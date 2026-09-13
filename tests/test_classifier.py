"""
Unit tests for ResNet-50 False Alarm Verification Engine.
"""
import pytest
from PIL import Image
from backend.config import RAW_IMAGES_DIR
from backend.core.classifier import SteelDefectVerifier

@pytest.fixture(scope="session")
def verifier_instance():
    verifier = SteelDefectVerifier()
    verifier.load_model()
    return verifier

def test_verifier_initialization(verifier_instance):
    assert verifier_instance._loaded is True
    assert verifier_instance.model is not None

def test_classify_crop(verifier_instance):
    img_path = RAW_IMAGES_DIR / "inclusion_1.jpg"
    img = Image.open(img_path).convert("RGB")
    res = verifier_instance.classify_crop(img)
    
    assert "top1_label" in res
    assert "top1_confidence" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 6
    # Probabilities should sum to approximately 1.0
    total_prob = sum(res["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.05

def test_verify_detection_consensus(verifier_instance):
    img_path = RAW_IMAGES_DIR / "patches_1.jpg"
    img = Image.open(img_path).convert("RGB")
    
    # Simulate a patch detection
    ver_res = verifier_instance.verify_detection(
        full_image=img,
        bbox=[40.0, 10.0, 105.0, 95.0],
        yolo_class="patches",
        yolo_conf=0.85
    )
    assert ver_res["verified"] is True
    assert "consensus_confidence" in ver_res
    assert "false_alarm_risk" in ver_res
