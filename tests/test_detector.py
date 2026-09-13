"""
Unit and integration tests for Core YOLOv8 Defect Detector.
"""
import pytest
import numpy as np
from PIL import Image
from backend.config import RAW_IMAGES_DIR
from backend.core.detector import SteelDefectDetector

@pytest.fixture(scope="session")
def detector_instance():
    """Initialize detector once for the test session to avoid repeated CUDA warmup."""
    return SteelDefectDetector()

def test_detector_initialization(detector_instance):
    assert detector_instance.model is not None
    assert detector_instance._warmup_done is True
    assert len(detector_instance.model.names) == 6

def test_detect_sample_scratches(detector_instance):
    """Test detection on authentic scratches benchmark image."""
    img_path = RAW_IMAGES_DIR / "scratches_1.jpg"
    assert img_path.exists(), "Sample file scratches_1.jpg missing"

    result = detector_instance.detect(img_path, conf_threshold=0.25)
    assert result["status"] == "success"
    assert result["defect_count"] > 0
    assert result["annotated_image_b64"] is not None
    
    # Verify defect attributes
    first_defect = result["defects"][0]
    assert first_defect["defect_class"] in detector_instance.model.names.values()
    assert 0.0 <= first_defect["confidence"] <= 1.0
    assert len(first_defect["bounding_box"]) == 4
    assert first_defect["area_pixels"] > 0
    assert "severity_score" in first_defect
    assert "root_cause" in first_defect

def test_detect_clean_strip(detector_instance):
    """Test that clean stainless steel strip returns zero defects and PRIME disposition."""
    clean_path = RAW_IMAGES_DIR / "defect_free_strip_1.jpg"
    assert clean_path.exists(), "Sample file defect_free_strip_1.jpg missing"

    result = detector_instance.detect(clean_path, conf_threshold=0.25)
    assert result["status"] == "success"
    assert result["defect_count"] == 0
    assert result["disposition"]["disposition_code"] == "PRIME"
    assert result["disposition"]["overall_severity"] == 0.0

def test_detect_numpy_and_pil_inputs(detector_instance):
    """Verify polymorphic input support across various dimensions and dtypes."""
    # Standard 3-channel uint8
    arr = np.random.randint(100, 200, (200, 200, 3), dtype=np.uint8)
    res_np = detector_instance.detect(arr, conf_threshold=0.5)
    assert res_np["status"] == "success"

    # 1-channel grayscale with shape (H, W, 1) - typical line-scan camera raw format
    arr_1ch = np.random.randint(100, 200, (200, 200, 1), dtype=np.uint8)
    res_1ch = detector_instance.detect(arr_1ch, conf_threshold=0.5)
    assert res_1ch["status"] == "success"

    # 2D grayscale with shape (H, W)
    arr_2d = np.random.randint(100, 200, (200, 200), dtype=np.uint8)
    res_2d = detector_instance.detect(arr_2d, conf_threshold=0.5)
    assert res_2d["status"] == "success"

    # Float32 normalized array in [0.0, 1.0]
    arr_float = np.random.uniform(0.3, 0.8, (150, 150, 3)).astype(np.float32)
    res_float = detector_instance.detect(arr_float, conf_threshold=0.5)
    assert res_float["status"] == "success"

    # RGBA image with alpha channel
    pil_rgba = Image.new("RGBA", (180, 180), (140, 140, 140, 255))
    res_rgba = detector_instance.detect(pil_rgba, conf_threshold=0.5)
    assert res_rgba["status"] == "success"

def test_detect_invalid_input(detector_instance):
    """Invalid image paths or corrupted data should raise ValueError."""
    with pytest.raises(Exception):
        detector_instance.detect("non_existent_file_xyz_123.jpg")
    with pytest.raises(Exception):
        detector_instance.detect(123456)  # integer is not a valid image input

def test_detect_with_false_alarm_verifier(detector_instance):
    """Verify detector with secondary ResNet verifier enabled."""
    clean_path = RAW_IMAGES_DIR / "defect_free_strip_1.jpg"
    res = detector_instance.detect(clean_path, conf_threshold=0.25, verify_false_alarms=True)
    assert res["status"] == "success"
    assert res["disposition"]["disposition_code"] == "PRIME"
    for d in res["defects"]:
        if d.get("is_potential_false_alarm"):
            assert d["severity_score"] == 0.0
            assert d["severity_tier"] == "Suppressed / False Alarm"


def test_detect_batch_chunks(detector_instance):
    """Test detect_batch with multiple frames spanning across chunk boundaries."""
    img_path = RAW_IMAGES_DIR / "scratches_1.jpg"
    assert img_path.exists(), "Sample file scratches_1.jpg missing"
    # Test batch of 7 images with chunk_size=4 (tests chunk 1 of 4, chunk 2 of 3)
    batch_inputs = [img_path] * 7
    result = detector_instance.detect_batch(batch_inputs, chunk_size=4)
    assert result["status"] == "success"
    assert result["total_frames_inspected"] == 7
    assert len(result["frame_summaries"]) == 7
    assert result["throughput_fps"] > 25.0
    assert "coil_overall_disposition" in result
    assert result["coil_overall_disposition"]["disposition_code"] in ["PRIME", "REWORK", "DOWNGRADE", "SCRAP"]
    # Check that each frame summary has expected fields
    for idx, fs in enumerate(result["frame_summaries"]):
        assert fs["frame_index"] == idx + 1
        assert "latency_ms" in fs
        assert "defect_count" in fs


def test_detect_multi_finish_clean_strips(detector_instance):
    """Test detect_batch and detect on clean strips across diverse surface finishes (0.0% FAR)."""
    clean_filenames = [
        "defect_free_strip_1.jpg",
        "defect_free_strip_2.jpg",
        "defect_free_strip_3_mirror.jpg",
        "defect_free_strip_4_brushed.jpg",
        "defect_free_strip_5_mill2b.jpg",
    ]
    existing_cleans = [RAW_IMAGES_DIR / fn for fn in clean_filenames if (RAW_IMAGES_DIR / fn).exists()]
    assert len(existing_cleans) == 5, f"Expected 5 clean strip finishes, found {len(existing_cleans)}"

    # Test individual single-frame detection
    for clean_path in existing_cleans:
        res = detector_instance.detect(clean_path, conf_threshold=0.25, verify_false_alarms=True)
        assert res["status"] == "success"
        active_defects = [d for d in res["defects"] if not d.get("is_potential_false_alarm", False)]
        assert len(active_defects) == 0, f"False alarm triggered on clean finish {clean_path.name}: {active_defects}"
        assert res["disposition"]["disposition_code"] == "PRIME"

    # Test batch detection on all 5 clean strip finishes together
    batch_res = detector_instance.detect_batch(existing_cleans, chunk_size=4)
    assert batch_res["status"] == "success"
    assert batch_res["total_frames_inspected"] == 5
    assert batch_res["total_defects_found"] == 0
    assert batch_res["coil_overall_disposition"]["disposition_code"] == "PRIME"

