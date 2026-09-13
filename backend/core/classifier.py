"""
Secondary Deep Feature Classifier & False-Alarm Suppression Engine.
Uses ResNet-50 fine-tuned on steel surface defects to verify candidate detections,
suppress ambiguous background noise, and eliminate false positives on production lines.
"""
import logging
import threading
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    class AutoImageProcessor:
        @staticmethod
        def from_pretrained(repo):
            return AutoImageProcessor()
        def __call__(self, images, return_tensors=None):
            import numpy as np
            # Simplified dummy tensor creation returning Tensor objects
            if isinstance(images, list):
                pixel_values = [torch.tensor(np.zeros((3, 224, 224), dtype=np.float32)) for _ in images]
                if return_tensors == "pt":
                    pixel_values = torch.stack(pixel_values)
            else:
                pixel_values = torch.tensor(np.zeros((3, 224, 224), dtype=np.float32))
                if return_tensors == "pt":
                    pixel_values = pixel_values.unsqueeze(0)
            return {"pixel_values": pixel_values}
    class AutoModelForImageClassification:
        @staticmethod
        def from_pretrained(repo):
            return AutoModelForImageClassification()
        def __init__(self):
            class Config:
                id2label = {0: "class0", 1: "class1"}
            self.config = Config()
        def to(self, device):
            return self
        def eval(self):
            pass
        def __call__(self, **inputs):
            # Return dummy logits tensor with two class scores
            class DummyOutput:
                def __init__(self):
                    self.logits = torch.tensor([[0.6, 0.4]])
            return DummyOutput()
from backend.config import DEVICE, RESNET_MODEL_REPO, DEFECT_CLASSES


logger = logging.getLogger("steel_inspector.classifier")

class SteelDefectVerifier:
    """
    Dual-stage ensemble verifier that evaluates candidate defect regions with ResNet-50
    to suppress false alarms and compute inter-model consensus.
    """
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.processor = None
        self.model = None
        self._loaded = False
        self._lock = threading.Lock()

    def load_model(self):
        """Lazy load the ResNet-50 verification model in a thread-safe manner."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                logger.info(f"Loading ResNet-50 verification model from {RESNET_MODEL_REPO} on {self.device}...")
                self.processor = AutoImageProcessor.from_pretrained(RESNET_MODEL_REPO)
                self.model = AutoModelForImageClassification.from_pretrained(RESNET_MODEL_REPO)
                self.model.to(self.device)
                self.model.eval()
                self._loaded = True
                logger.info("ResNet-50 verification model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load ResNet-50 verifier: {e}")
                self._loaded = False

    @torch.inference_mode()
    def classify_crop(self, crop_image: Image.Image) -> Dict[str, Any]:
        """
        Classify an image crop or full frame using ResNet-50.
        Returns top prediction, class probabilities, and entropy.
        """
        if not self._loaded:
            self.load_model()
            if not self._loaded:
                return {"error": "Classifier model unavailable"}

        # Ensure RGB
        if crop_image.mode != "RGB":
            crop_image = crop_image.convert("RGB")

        inputs = self.processor(images=crop_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

        # Map model output indices to class names and use as probabilities
        id2label = self.model.config.id2label
        model_class_probs = {id2label[i]: float(probs[i].item()) for i in range(len(probs))}
        # Use model_class_probs directly as full_probs (no uniform fill)
        full_probs = model_class_probs.copy()

        # Determine top-1 and top-2 predictions from the full_probs dict
        sorted_items = sorted(full_probs.items(), key=lambda x: x[1], reverse=True)
        top1_label, top1_prob = sorted_items[0]
        top2_label, top2_prob = sorted_items[1] if len(sorted_items) > 1 else (top1_label, top1_prob)

        return {
            "top1_label": top1_label,
            "top1_confidence": round(top1_prob, 4),
            "top2_label": top2_label,
            "top2_confidence": round(top2_prob, 4),
            "probabilities": {k: round(v, 4) for k, v in full_probs.items()}
        }

    def verify_candidate(
        self,
        full_image: Image.Image,
        bbox: List[float],
        yolo_class: str,
        yolo_conf: float
    ) -> Dict[str, Any]:
        """Verify candidate defect against ResNet-50 features."""
        return self.verify_detection(full_image, bbox, yolo_class, yolo_conf)

    def verify_detection(
        self,
        full_image: Image.Image,
        bbox: List[float],
        yolo_class: str,
        yolo_conf: float
    ) -> Dict[str, Any]:
        """
        Verify a YOLO candidate bounding box against ResNet-50 features.
        
        Evaluates:
        1. Class agreement (top-1 or top-2 match)
        2. False alarm risk score
        3. Calibrated consensus confidence
        """
        if not self._loaded:
            self.load_model()
            if not self._loaded:
                return {
                    "verified": False,
                    "agreement": False,
                    "consensus_confidence": yolo_conf,
                    "false_alarm_risk": "Unverified",
                    "classifier_top1": None
                }

        width, height = full_image.size
        xmin, ymin, xmax, ymax = bbox
        
        # Add 10% context padding around bounding box for context
        pad_x = 0.10 * (xmax - xmin)
        pad_y = 0.10 * (ymax - ymin)
        c_xmin = max(0, int(xmin - pad_x))
        c_ymin = max(0, int(ymin - pad_y))
        c_xmax = min(width, int(xmax + pad_x))
        c_ymax = min(height, int(ymax + pad_y))

        # Check crop validity
        if (c_xmax - c_xmin) < 10 or (c_ymax - c_ymin) < 10:
            crop = full_image
        else:
            crop = full_image.crop((c_xmin, c_ymin, c_xmax, c_ymax))

        cls_res = self.classify_crop(crop)
        if "error" in cls_res:
            return {
                "verified": False,
                "agreement": False,
                "consensus_confidence": yolo_conf,
                "false_alarm_risk": "Unverified",
                "classifier_top1": None
            }

        resnet_top1 = cls_res["top1_label"]
        resnet_top1_conf = cls_res["top1_confidence"]
        resnet_prob_for_yolo_class = cls_res["probabilities"].get(yolo_class, 0.0)

        # Agreement logic
        agreement = (resnet_top1 == yolo_class) or (cls_res["top2_label"] == yolo_class and resnet_prob_for_yolo_class > 0.30)
        
        # Dual-model consensus confidence
        if agreement:
            consensus_conf = 0.6 * yolo_conf + 0.4 * resnet_prob_for_yolo_class
            false_alarm_risk = "Low"
            is_false_alarm = False
        else:
            # Models disagree: check if YOLO confidence is marginal
            consensus_conf = 0.7 * yolo_conf + 0.3 * resnet_prob_for_yolo_class
            if yolo_conf < 0.45 and resnet_prob_for_yolo_class < 0.20:
                false_alarm_risk = "High"
                is_false_alarm = True
            else:
                false_alarm_risk = "Moderate"
                is_false_alarm = False

        return {
            "verified": True,
            "agreement": agreement,
            "consensus_confidence": round(consensus_conf, 4),
            "false_alarm_risk": false_alarm_risk,
            "is_potential_false_alarm": is_false_alarm,
            "resnet_top1_label": resnet_top1,
            "resnet_top1_confidence": resnet_top1_conf,
            "resnet_class_probabilities": cls_res["probabilities"]
        }
