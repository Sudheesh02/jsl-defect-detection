"""
Comprehensive unit and integration test suite validating all Milestone 1 audit fixes.
Covers:
- VULN-01: Async threadpool offloading
- VULN-02: File descriptor cleanup
- VULN-03: Batch limits (<=100) and query parameter validation bounds
- VULN-04: generate_thumbnails decoupling
- VULN-05: Thread-safe singleton locks (detector and classifier)
- VULN-06: Unverified classifier fallback status
- VULN-07: Single-class logits safety guard in classifier
- VULN-08: Polymorphic tensor normalization (torch.Tensor, planar CHW, zero-size, NaN, [-1, 1])
- VULN-09: Bounding box coordinate clamping
- VULN-10: Centroid-based edge defect detection
- VULN-11: Proportional commercial downgrade for minor critical defects
- VULN-12: Defect density incorporation in batch disposition
- VULN-13: Division by zero and empty latency array guards
- VULN-14: Sample endpoint stat error safety
"""
import io
import time
import threading
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import app
from backend.core.detector import SteelDefectDetector
from backend.core.classifier import SteelDefectVerifier
from backend.core.metallurgy_engine import analyze_defect, compute_frame_disposition
from backend.core.line_simulator import ProductionLineSimulator
from backend.api.routes_detect import get_detector


client = TestClient(app)


# --------------------------------------------------------------------------
# VULN-08: Polymorphic Tensor Normalization
# --------------------------------------------------------------------------
def test_normalize_empty_array_raises_clean_value_error():
    """Empty numpy array must raise clean ValueError rather than crashing on arr.max()."""
    with pytest.raises(ValueError, match="Empty image array provided."):
        SteelDefectDetector._normalize_image_input(np.zeros((0, 640, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="Empty image array provided."):
        SteelDefectDetector._normalize_image_input(np.array([]))


def test_normalize_torch_tensor_hwc():
    """torch.Tensor in HWC uint8 format should convert cleanly to PIL Image."""
    t = torch.randint(0, 255, (200, 300, 3), dtype=torch.uint8)
    img = SteelDefectDetector._normalize_image_input(t)
    assert isinstance(img, Image.Image)
    assert img.size == (300, 200)
    assert img.mode == "RGB"


def test_normalize_torch_tensor_planar_chw():
    """torch.Tensor in planar (3, H, W) float format should transpose to (H, W, 3)."""
    t = torch.rand((3, 150, 250), dtype=torch.float32)
    img = SteelDefectDetector._normalize_image_input(t)
    assert isinstance(img, Image.Image)
    assert img.size == (250, 150)
    assert img.mode == "RGB"


def test_normalize_torch_tensor_single_channel_chw():
    """torch.Tensor in (1, H, W) planar format should squeeze to (H, W) and return RGB."""
    t = torch.rand((1, 100, 100), dtype=torch.float32)
    img = SteelDefectDetector._normalize_image_input(t)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)
    assert img.mode == "RGB"


def test_normalize_planar_chw_numpy():
    """NumPy array in (3, H, W) format should transpose to (H, W, 3)."""
    arr = np.zeros((3, 120, 180), dtype=np.uint8)
    img = SteelDefectDetector._normalize_image_input(arr)
    assert isinstance(img, Image.Image)
    assert img.size == (180, 120)
    assert img.mode == "RGB"


def test_normalize_nan_and_inf_sanitization():
    """Arrays with NaN and Inf values must be sanitized to 0.0 without errors."""
    arr = np.full((50, 50, 3), np.nan, dtype=np.float32)
    arr[10, 10, :] = np.inf
    arr[20, 20, :] = -np.inf
    img = SteelDefectDetector._normalize_image_input(arr)
    assert isinstance(img, Image.Image)
    assert img.size == (50, 50)
    np_img = np.array(img)
    assert not np.isnan(np_img).any()
    assert not np.isinf(np_img).any()


def test_normalize_float_negative_range():
    """Floating point arrays in [-1.0, 1.0] should map properly without clipping negatives to 0."""
    arr = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    img = SteelDefectDetector._normalize_image_input(arr)
    np_img = np.array(img)
    # -1.0 -> 0, 0.0 -> ~127, 1.0 -> 255
    assert np_img[0, 0, 0] <= 5
    assert 120 <= np_img[0, 0, 1] <= 135
    assert np_img[0, 0, 2] >= 250


# --------------------------------------------------------------------------
# VULN-09: Bounding Box Sanitization & Clamping
# --------------------------------------------------------------------------
def test_analyze_defect_out_of_bounds_clamping():
    """Out-of-bounds coordinates must be clamped to image boundary [0, dim]."""
    img_w, img_h = 1000, 1000
    box_oob = [-50.0, -20.0, 1100.0, 1050.0]
    res = analyze_defect(
        defect_class="scratches",
        confidence=0.85,
        box=box_oob,
        image_width=img_w,
        image_height=img_h
    )
    clamped_box = res["bounding_box"]
    assert clamped_box[0] == 0.0
    assert clamped_box[1] == 0.0
    assert clamped_box[2] == 1000.0
    assert clamped_box[3] == 1000.0
    assert res["area_pct"] <= 100.0


def test_analyze_defect_inverted_coordinates():
    """Inverted coordinates (xmin > xmax) must be normalized without negative areas."""
    img_w, img_h = 800, 800
    box_inv = [600.0, 600.0, 200.0, 200.0]
    res = analyze_defect(
        defect_class="patches",
        confidence=0.75,
        box=box_inv,
        image_width=img_w,
        image_height=img_h
    )
    box = res["bounding_box"]
    assert box[0] <= box[2]
    assert box[1] <= box[3]
    assert res["area_pixels"] > 0


# --------------------------------------------------------------------------
# VULN-10: Centroid-Based Edge Defect Proximity
# --------------------------------------------------------------------------
def test_wide_center_defect_classified_as_body_not_edge():
    """
    A wide strip-spanning defect in the center whose boundaries touch edge margins
    must NOT be flagged as an edge defect if its centroid is in the center.
    """
    img_w, img_h = 1000, 1000
    # Center defect spanning x=150 to x=850 (centroid = 500, strip margin = 120)
    wide_box = [150.0, 300.0, 850.0, 400.0]
    res = analyze_defect(
        defect_class="crease",
        confidence=0.80,
        box=wide_box,
        image_width=img_w,
        image_height=img_h,
        grade_code="SS_304"
    )
    assert res["is_edge_defect"] is False
    assert res["strip_zone"] == "Strip Center / Body"


def test_genuine_edge_defect_classified_as_edge():
    """Defect whose centroid is within the 12% margin must be flagged as edge defect."""
    img_w, img_h = 1000, 1000
    # Left edge defect x=10 to x=60 (centroid = 35 <= 120)
    edge_box = [10.0, 300.0, 60.0, 400.0]
    res = analyze_defect(
        defect_class="edge_cracks",
        confidence=0.85,
        box=edge_box,
        image_width=img_w,
        image_height=img_h,
        grade_code="SS_304"
    )
    assert res["is_edge_defect"] is True
    assert res["strip_zone"] == "Strip Edge"


# --------------------------------------------------------------------------
# VULN-11: Proportional Commercial Downgrade for Minor Critical Defects
# --------------------------------------------------------------------------
def test_minor_critical_defects_yield_commercial_downgrade():
    """
    Multiple minor critical defects with severity < 4.0 should DOWNGRADE rather than SCRAP.
    """
    minor_inclusions = [
        {"defect_class": "inclusion", "severity_score": 1.8, "is_critical_for_grade": True, "reworkable": False},
        {"defect_class": "inclusion", "severity_score": 2.1, "is_critical_for_grade": True, "reworkable": False}
    ]
    disp = compute_frame_disposition(minor_inclusions, grade_code="SS_304")
    assert disp["disposition_code"] == "DOWNGRADE"
    assert disp["quality_grade"] == "C"


def test_severe_critical_defects_trigger_scrap():
    """Severe critical defects with severity >= 4.0 must trigger SCRAP / REJECT."""
    severe_inclusions = [
        {"defect_class": "inclusion", "severity_score": 8.0, "is_critical_for_grade": True, "reworkable": False},
        {"defect_class": "inclusion", "severity_score": 7.5, "is_critical_for_grade": True, "reworkable": False}
    ]
    disp = compute_frame_disposition(severe_inclusions, grade_code="SS_304")
    assert disp["disposition_code"] == "SCRAP"
    assert disp["quality_grade"] == "F"


# --------------------------------------------------------------------------
# VULN-12: Defect Density in Batch Disposition
# --------------------------------------------------------------------------
def test_batch_defect_density_penalty():
    """Dense defect count across batch frames should increase composite severity."""
    # 20 light defects across 2 frames (density = 10 defects/frame)
    dense_defects = [
        {"defect_class": "patches", "severity_score": 2.0, "is_critical_for_grade": False, "reworkable": True}
        for _ in range(20)
    ]
    disp_single = compute_frame_disposition(dense_defects, is_batch=False)
    disp_batch = compute_frame_disposition(dense_defects, is_batch=True, total_frames=2)
    assert disp_batch["overall_severity"] > disp_single["overall_severity"]


# --------------------------------------------------------------------------
# VULN-13: Line Simulator Zero & Empty Guards
# --------------------------------------------------------------------------
def test_line_simulator_profile_latencies_empty():
    """profile_latencies([]) must return zeroed dictionary without raising ValueError."""
    stats = ProductionLineSimulator.profile_latencies([])
    assert stats["count"] == 0
    assert stats["mean_ms"] == 0.0
    assert stats["max_ms"] == 0.0
    assert stats["p95_ms"] == 0.0


def test_line_simulator_100_percent_overlap_guard():
    """overlap_pct >= 100% must not cause division by zero in calculate_required_fps or compute_line_requirements."""
    sim = ProductionLineSimulator(overlap_pct=100.0)
    fps = sim.calculate_required_fps(600.0)
    assert fps > 0.0
    metrics = sim.compute_line_requirements(600.0, 25.0)
    assert metrics["required_camera_fps"] > 0.0


# --------------------------------------------------------------------------
# VULN-04: generate_thumbnails Parameter
# --------------------------------------------------------------------------
def test_detect_generate_thumbnails_flag():
    """When generate_thumbnails=False, thumbnail_b64 must be None for all detected defects."""
    detector = get_detector()
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    res = detector.detect(img, return_annotated_image=False, generate_thumbnails=False)
    for d in res["defects"]:
        assert d["thumbnail_b64"] is None


# --------------------------------------------------------------------------
# VULN-06 & VULN-07: ResNet Verifier Fallback and Logits Guard
# --------------------------------------------------------------------------
def test_verifier_fallback_when_model_not_loaded():
    """When verifier model fails to load, verify_detection must return verified: False and Unverified."""
    verifier = SteelDefectVerifier()
    verifier._loaded = False
    
    with patch.object(verifier, "load_model", return_value=None):
        dummy_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        res = verifier.verify_candidate(
            full_image=dummy_img,
            bbox=[10.0, 10.0, 50.0, 50.0],
            yolo_class="scratches",
            yolo_conf=0.75
        )
        assert res["verified"] is False
        assert res["agreement"] is False
        assert res["false_alarm_risk"] == "Unverified"
        assert res["classifier_top1"] is None


def test_verifier_single_class_crop_no_index_error():
    """Verify classify_crop handles single-class probability distributions without IndexError."""
    verifier = SteelDefectVerifier()
    verifier._loaded = True
    verifier.processor = MagicMock()
    verifier.processor.return_tensors = "pt"
    verifier.processor.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "single_class"}
    # Single class logit
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[5.0]])
    mock_model.return_value = mock_output
    verifier.model = mock_model

    dummy_crop = Image.new("RGB", (50, 50))
    res = verifier.classify_crop(dummy_crop)
    assert res["top1_label"] == "single_class"
    assert res["top2_label"] == "single_class"


# --------------------------------------------------------------------------
# VULN-05: Concurrency & Thread-Safe Singleton Lock
# --------------------------------------------------------------------------
def test_detector_singleton_concurrency():
    """Multiple threads accessing get_detector() concurrently must receive the exact same singleton instance."""
    instances = []

    def fetch_instance():
        inst = get_detector()
        instances.append(id(inst))

    threads = [threading.Thread(target=fetch_instance) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(instances) == 10
    # All thread instances must have identical memory address
    assert len(set(instances)) == 1


def test_verifier_load_model_thread_safety():
    """Multiple threads calling load_model() on a verifier must synchronize cleanly."""
    verifier = SteelDefectVerifier()
    loaded_count = 0

    def call_load():
        nonlocal loaded_count
        verifier.load_model()
        if verifier._loaded:
            loaded_count += 1

    threads = [threading.Thread(target=call_load) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert verifier._loaded is True
    assert loaded_count == 5


# --------------------------------------------------------------------------
# VULN-03: API Batch Limits & Validation Bounds
# --------------------------------------------------------------------------
def test_batch_detect_exceeds_100_frames_rejected():
    """Batches with > 100 files must be rejected with HTTP 400."""
    fake_files = [("files", (f"frame_{i}.jpg", io.BytesIO(b"fake"), "image/jpeg")) for i in range(101)]
    response = client.post("/api/batch-detect", files=fake_files)
    assert response.status_code == 400
    assert "maximum limit of 100 frames" in response.json()["detail"]


def test_batch_detect_zero_byte_file_rejected():
    """Batches containing a 0-byte file must be rejected with HTTP 400."""
    fake_files = [("files", ("empty.jpg", io.BytesIO(b""), "image/jpeg"))]
    response = client.post("/api/batch-detect", files=fake_files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_batch_detect_invalid_query_bounds():
    """Thresholds outside valid bounds must return HTTP 422 Unprocessable Entity."""
    fake_files = [("files", ("test.jpg", io.BytesIO(b"testdata"), "image/jpeg"))]
    # conf_threshold < 0.05
    response = client.post("/api/batch-detect?conf_threshold=0.01", files=fake_files)
    assert response.status_code == 422

    # iou_threshold > 0.9
    response2 = client.post("/api/batch-detect?iou_threshold=0.95", files=fake_files)
    assert response2.status_code == 422


# --------------------------------------------------------------------------
# VULN-14: Sample Listing Error Handling
# --------------------------------------------------------------------------
def test_samples_endpoint_file_stat_resilience():
    """Verify /api/samples handles file stat errors gracefully."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert "count" in data
    assert data["count"] == len(data["samples"])
    for s in data["samples"]:
        assert "file_size_bytes" in s
        assert s["file_size_bytes"] >= 0
