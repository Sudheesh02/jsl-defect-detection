"""
Configuration settings for Jindal Stainless AI Defect Detection Backend.
"""
import os
from pathlib import Path
try:
    import torch
except ImportError:
    # Minimal torch stub for environments without PyTorch
    class _DummyCuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def get_device_name(idx=0):
            return "CPU"
    class _DummyCudnn:
        allow_tf32 = False
        benchmark = False
    class _DummyMatmul:
        allow_tf32 = False
    class _DummyBackends:
        cudnn = _DummyCudnn()
        cuda = type('Cuda', (), {'matmul': _DummyMatmul()})
    class _DummyTensor:
        pass
    class _DummyTorch:
        cuda = _DummyCuda()
        backends = _DummyBackends()
        @staticmethod
        def inference_mode():
            class _DummyCtx:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False
            return _DummyCtx()
        Tensor = _DummyTensor
        @staticmethod
        def from_numpy(arr):
            return arr
    torch = _DummyTorch()


# Enable TensorFloat-32 (TF32) for CUDA devices for faster matrix ops while preserving FP32 accuracy.
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
RAW_IMAGES_DIR = SAMPLE_DATA_DIR / "raw_images"
ANNOTATIONS_DIR = SAMPLE_DATA_DIR / "annotations"

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"
HALF_PRECISION = True if DEVICE == "cuda" else False

# Model Configuration
YOLO_MODEL_REPO = "kirkdokizelli/steel-defect-yolov8"
YOLO_MODEL_FILE = "best.pt"
RESNET_MODEL_REPO = "vectorsight/steel-defects-classifier-resnet"

# Cache Directories
HF_CACHE_DIR = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface" / "hub"))

# Defect Classes (NEU-DET Benchmark standard)
DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Industrial Visualization Color Palette (RGB tuples & Hex strings)
CLASS_COLORS = {
    "crazing": {"rgb": (255, 128, 0), "hex": "#ff8000"},          # Neon Orange (Thermal stress)
    "inclusion": {"rgb": (0, 220, 255), "hex": "#00dcff"},        # Cyan (Smelting slag)
    "patches": {"rgb": (0, 200, 150), "hex": "#00c896"},          # Teal (Roll slippage)
    "pitted_surface": {"rgb": (180, 70, 255), "hex": "#b446ff"},  # Purple (Pitting corrosion)
    "rolled-in_scale": {"rgb": (255, 50, 80), "hex": "#ff3250"},  # Crimson (Hot roll scale)
    "scratches": {"rgb": (255, 210, 0), "hex": "#ffd200"},        # Amber (Mechanical guide abrasion)
}

# Production Line Defaults
DEFAULT_LINE_SPEED_M_PER_MIN = 600.0  # 10 m/s (standard cold rolling / finishing speed)
DEFAULT_CAMERA_FOV_MM = 1250.0        # Typical steel strip width: 1250mm
DEFAULT_CAMERA_PIXELS_X = 2048        # 2K line scan camera
DEFAULT_CAMERA_PIXELS_Y = 2048
MM_PER_PIXEL = DEFAULT_CAMERA_FOV_MM / DEFAULT_CAMERA_PIXELS_X  # ~0.61 mm/pixel

# Inference Defaults
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640

# Batch Processing & Hardware Optimization Constants
DEFAULT_BATCH_CHUNK_SIZE = 4   # Optimal chunk size for RTX 5060 (>58 FPS, <100ms batch latency)
MAX_BATCH_CHUNK_SIZE = 8
GPU_WARMUP_ITERATIONS = 8      # Number of warmup cycles to lock GPU in P0 boost power state

# Disposition Severity Thresholds
DISPOSITION_THRESHOLDS = {
    "prime_max_severity": 1.5,
    "rework_max_severity": 4.5,
    "downgrade_max_severity": 7.5,
    # Above 7.5 -> SCRAP / REJECT
}
