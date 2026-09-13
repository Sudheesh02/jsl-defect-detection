import torch

class DummyResult:
    """Mimic the result object returned by ultralytics YOLO inference.
    Provides a ``boxes`` attribute containing a single dummy detection.
    """
    class _Box:
        def __init__(self, img_width, img_height):
            # Dummy bounding box coordinates as torch tensor: [xmin, ymin, xmax, ymax]
            self.xyxy = torch.tensor([[img_width * 0.1, img_height * 0.1, img_width * 0.5, img_height * 0.5]])
            # Dummy confidence scores tensor
            self.conf = torch.tensor([0.9])
            # Dummy class indices tensor (using first defect class index 0)
            self.cls = torch.tensor([0])
    def __init__(self, img_width, img_height):
        # Provide a list with a single dummy detection box.
        self.boxes = [self._Box(img_width, img_height)]

class YOLO:
    """Very small stub that mimics ``ultralytics.YOLO``.
    It stores the model path but does not actually load any weights.
    The callable interface ``model(image, conf=..., iou=..., device=...)``
    returns a list with a single ``DummyResult`` instance.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Provide a dummy mapping from class index to defect class name.
        try:
            from backend.config import DEFECT_CLASSES
            self.names = {i: name for i, name in enumerate(DEFECT_CLASSES)}
        except Exception:
            # Fallback to generic names if config cannot be imported.
            self.names = {i: f"class_{i}" for i in range(10)}

    def __call__(self, img, conf: float = 0.25, iou: float = 0.45, device: str = "cpu", verbose: bool = False, **kwargs):
        if hasattr(img, "size"):
            width, height = img.size
        else:
            import numpy as np
            arr = np.asarray(img)
            if arr.ndim == 3:
                if arr.shape[0] in (1, 3):
                    _, height, width = arr.shape
                else:
                    height, width, _ = arr.shape
            elif arr.ndim == 2:
                height, width = arr.shape
            else:
                raise ValueError("Unsupported image shape for YOLO stub")
        return [DummyResult(width, height)]

    def to(self, device: str):
        return self

    @property
    def model(self):
        class _Model:
            @staticmethod
            def half():
                return None
        return _Model()

    def __repr__(self):
        return f"<Stub YOLO model path={self.model_path}>"

__all__ = ["YOLO"]
