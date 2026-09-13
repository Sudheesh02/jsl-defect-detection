"""
Empirical Stress Testing Harness - Challenger 1 (Concurrency & Tensor Challenger).
Milestone 1 Robustness & Failure-Mode Verification for Jindal Stainless Defect Platform.

Adversarial Stress Scenarios:
1. Multithreaded concurrent initialization of `get_detector()` and `_normalize_image_input()`.
2. Extreme tensor inputs: empty arrays, 1D arrays, planar CHW arrays (3, H, W) and (1, H, W),
   PyTorch torch.Tensor objects (CPU, CUDA, autograd, non-contiguous, half-precision),
   arrays with NaNs and Infs, negative float arrays [-1.0, 1.0], large floats.
3. Out-of-bounds bounding boxes ([-50, -50, 1500, 1500]), inverted boxes ([400, 400, 100, 100]),
   degenerate boxes, completely off-screen boxes passed to `analyze_defect`.
4. Line simulator latency profiling on empty lists, 1-element lists, and 100% overlap kinematics.
"""
import io
import time
import threading
import concurrent.futures
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import pytest
import torch

from backend.core.detector import SteelDefectDetector
from backend.core.classifier import SteelDefectVerifier
from backend.core.metallurgy_engine import (
    analyze_defect,
    compute_frame_disposition,
    EXTENDED_DEFECT_TAXONOMY
)
from backend.core.line_simulator import ProductionLineSimulator
import backend.api.routes_detect as routes_detect


# ==============================================================================
# SECTION 1: Multithreaded Concurrent Stress Tests
# ==============================================================================

class TestConcurrencyStress:
    """Stress-test thread-safety, race conditions, and synchronization locks."""

    def test_concurrent_get_detector_cold_race(self):
        """
        Simulate 30 threads simultaneously racing into cold uninitialized `get_detector()`.
        Verifies that only a single instance is ever created and no race conditions occur.
        """
        # Reset the global singleton detector to None under lock to test cold race
        with routes_detect._detector_lock:
            routes_detect._detector = None

        barrier = threading.Barrier(30)
        results = []
        errors = []

        def worker():
            try:
                # Synchronize all 30 threads to call get_detector at the exact same instant
                barrier.wait(timeout=10.0)
                detector = routes_detect.get_detector()
                results.append(id(detector))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert len(errors) == 0, f"Encountered errors in concurrent initialization: {errors}"
        assert len(results) == 30, f"Expected 30 results, got {len(results)}"
        # All 30 threads must have returned the exact identical instance id
        assert len(set(results)) == 1, f"Multiple instances instantiated! IDs: {set(results)}"
        assert routes_detect._detector is not None

    def test_concurrent_image_normalization_heavy_traffic(self):
        """
        Stress test `_normalize_image_input` under high concurrency (40 parallel workers,
        50 iterations each = 2,000 total normalizations across heterogeneous inputs).
        """
        sample_inputs = [
            # 1. Standard HWC uint8
            np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8),
            # 2. Planar CHW float32
            np.random.rand(3, 100, 140).astype(np.float32),
            # 3. Single-channel planar CHW float32
            np.random.rand(1, 150, 150).astype(np.float32),
            # 4. Normalized float [-1.0, 1.0]
            (np.random.rand(80, 80, 3).astype(np.float32) * 2.0) - 1.0,
            # 5. Grayscale 2D uint8
            np.random.randint(0, 256, (200, 200), dtype=np.uint8),
            # 6. Torch tensor HWC uint8
            torch.randint(0, 256, (100, 100, 3), dtype=torch.uint8),
            # 7. Torch tensor planar CHW float32
            torch.rand((3, 120, 120), dtype=torch.float32),
            # 8. Array with NaNs and Infs
            np.array([[[np.nan, np.inf, -np.inf]] * 40] * 40, dtype=np.float32),
            # 9. PIL Image
            Image.new("RGB", (64, 64), color=(128, 128, 128)),
            # 10. Raw JPEG bytes
            None  # Will generate below
        ]
        # Generate valid JPEG bytes
        buf = io.BytesIO()
        Image.new("RGB", (50, 50), color=(50, 100, 150)).save(buf, format="JPEG")
        sample_inputs[9] = buf.getvalue()

        def normalize_task(idx):
            inp = sample_inputs[idx % len(sample_inputs)]
            # If tensor, clone to ensure fresh copy
            if isinstance(inp, torch.Tensor):
                inp = inp.clone()
            elif isinstance(inp, np.ndarray):
                inp = inp.copy()
            res = SteelDefectDetector._normalize_image_input(inp)
            assert isinstance(res, Image.Image)
            assert res.mode == "RGB"
            assert res.width > 0 and res.height > 0
            return res.size

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(normalize_task, i) for i in range(2000)]
            completed, uncompleted = concurrent.futures.wait(futures, timeout=60.0)

        assert len(uncompleted) == 0, f"Timeout waiting for normalization tasks: {len(uncompleted)} remaining"
        for f in completed:
            assert f.exception() is None, f"Worker raised exception: {f.exception()}"

    def test_concurrent_normalization_error_handling(self):
        """
        Ensure concurrent threads encountering invalid inputs (empty, 1D, 4D) cleanly raise
        ValueError without deadlocking or interfering with each other.
        """
        invalid_inputs = [
            np.zeros((0, 640, 3), dtype=np.uint8),
            np.array([]),
            np.zeros((100,)),
            np.zeros((2, 3, 64, 64)),
            "non_existent_file_path_xyz_123.jpg"
        ]

        errors_caught = []

        def worker(item):
            try:
                SteelDefectDetector._normalize_image_input(item)
            except (ValueError, FileNotFoundError, OSError) as e:
                errors_caught.append(type(e).__name__)

        threads = [threading.Thread(target=worker, args=(invalid_inputs[i % len(invalid_inputs)],)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors_caught) == 100


# ==============================================================================
# SECTION 2: Extreme Tensor Inputs
# ==============================================================================

class TestExtremeTensorInputs:
    """Stress-test tensor parser against adversarial, degenerate, and extreme camera streams."""

    @pytest.mark.parametrize("empty_shape", [
        (0,),
        (0, 640, 3),
        (640, 0, 3),
        (0, 0, 0),
        (0, 100),
        (100, 0)
    ])
    def test_empty_numpy_arrays_raise_clean_value_error(self, empty_shape):
        """All variations of 0-dimension or 0-size NumPy arrays must raise ValueError."""
        arr = np.zeros(empty_shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="Empty image array provided."):
            SteelDefectDetector._normalize_image_input(arr)

    def test_empty_torch_tensors_raise_clean_value_error(self):
        """Torch empty tensors must raise clean ValueError."""
        t1 = torch.empty((0, 3, 256, 256))
        with pytest.raises(ValueError, match="Empty image array provided."):
            SteelDefectDetector._normalize_image_input(t1)

        t2 = torch.zeros((0,))
        with pytest.raises(ValueError, match="Empty image array provided."):
            SteelDefectDetector._normalize_image_input(t2)

    def test_1d_arrays_raise_value_error(self):
        """1D arrays must raise ValueError indicating unsupported dimensions."""
        arr = np.zeros((256,), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported numpy array dimensions"):
            SteelDefectDetector._normalize_image_input(arr)

        t_1d = torch.ones((512,))
        with pytest.raises(ValueError, match="Unsupported numpy array dimensions"):
            SteelDefectDetector._normalize_image_input(t_1d)

    def test_planar_chw_variations(self):
        """Verify planar CHW arrays (3, H, W) and (1, H, W) across types and edge sizes."""
        # 1. Standard (3, H, W) float
        arr3 = np.random.rand(3, 480, 640).astype(np.float32)
        img3 = SteelDefectDetector._normalize_image_input(arr3)
        assert img3.size == (640, 480)
        assert img3.mode == "RGB"

        # 2. Standard (1, H, W) float
        arr1 = np.random.rand(1, 480, 640).astype(np.float32)
        img1 = SteelDefectDetector._normalize_image_input(arr1)
        assert img1.size == (640, 480)
        assert img1.mode == "RGB"

        # 3. Minimal planar (3, 1, 1)
        min_arr3 = np.ones((3, 1, 1), dtype=np.uint8)
        # Note: shape (3, 1, 1) has shape[0]=3, shape[2]=1 (in (1,3)), so not transposed, but squeezed to (3,1)
        min_img = SteelDefectDetector._normalize_image_input(min_arr3)
        assert min_img.mode == "RGB"

        # 4. Planar (3, 3, 640) where H=3, W=640
        arr_h3 = np.random.rand(3, 3, 640).astype(np.float32)
        img_h3 = SteelDefectDetector._normalize_image_input(arr_h3)
        assert img_h3.size == (640, 3)

    def test_torch_tensor_autograd(self):
        """Tensor with requires_grad=True should detach cleanly."""
        t_grad = torch.rand((3, 64, 64), requires_grad=True)
        img_grad = SteelDefectDetector._normalize_image_input(t_grad)
        assert img_grad.size == (64, 64)

    def test_torch_tensor_float16(self):
        """Float16 tensor should convert cleanly."""
        t_f16 = torch.rand((64, 64, 3), dtype=torch.float16)
        img_f16 = SteelDefectDetector._normalize_image_input(t_f16)
        assert img_f16.size == (64, 64)

    def test_torch_tensor_bfloat16(self):
        """Adversarial stress test: BFloat16 tensor ingestion."""
        t_bf16 = torch.rand((64, 64, 3), dtype=torch.bfloat16)
        img_bf16 = SteelDefectDetector._normalize_image_input(t_bf16)
        assert img_bf16.size == (64, 64)

    def test_torch_tensor_int64(self):
        """Int64 tensor should convert cleanly."""
        t_i64 = torch.randint(0, 255, (64, 64, 3), dtype=torch.int64)
        img_i64 = SteelDefectDetector._normalize_image_input(t_i64)
        assert img_i64.size == (64, 64)

    def test_torch_tensor_non_contiguous(self):
        """Non-contiguous tensor (strided slice) should convert cleanly."""
        t_orig = torch.rand((64, 128, 3))
        t_strided = t_orig[:, ::2, :]
        assert not t_strided.is_contiguous()
        img_strided = SteelDefectDetector._normalize_image_input(t_strided)
        assert img_strided.size == (64, 64)

    def test_torch_tensor_cuda(self):
        """CUDA tensor if available should convert cleanly."""
        if torch.cuda.is_available():
            t_cuda = torch.rand((3, 80, 80), device="cuda")
            img_cuda = SteelDefectDetector._normalize_image_input(t_cuda)
            assert img_cuda.size == (80, 80)
            assert img_cuda.mode == "RGB"

    def test_nans_and_infs_extreme_patterns(self):
        """Stress-test various patterns of NaNs and positive/negative Infs."""
        # 1. All NaNs
        arr_nan = np.full((50, 50, 3), np.nan, dtype=np.float32)
        img_nan = SteelDefectDetector._normalize_image_input(arr_nan)
        arr_out = np.array(img_nan)
        assert not np.isnan(arr_out).any()
        assert (arr_out == 0).all()

        # 2. All Infs
        arr_inf = np.full((50, 50, 3), np.inf, dtype=np.float32)
        img_inf = SteelDefectDetector._normalize_image_input(arr_inf)
        arr_out_inf = np.array(img_inf)
        assert not np.isinf(arr_out_inf).any()

        # 3. Mixed NaNs, +Infs, -Infs and valid floats
        arr_mix = np.random.uniform(-1.0, 1.0, (60, 60, 3)).astype(np.float32)
        arr_mix[0:20, :, :] = np.nan
        arr_mix[20:40, :, :] = np.inf
        arr_mix[40:60, :, :] = -np.inf
        img_mix = SteelDefectDetector._normalize_image_input(arr_mix)
        assert img_mix.size == (60, 60)
        res_np = np.array(img_mix)
        assert res_np.min() >= 0
        assert res_np.max() <= 255

    def test_negative_and_extreme_float_ranges(self):
        """Stress-test float scaling across unusual dynamic ranges."""
        # 1. Range [-1.0, 1.0] exactly
        arr_neg1_1 = np.linspace(-1.0, 1.0, 10000).reshape((100, 100, 1)).astype(np.float32)
        img1 = SteelDefectDetector._normalize_image_input(arr_neg1_1)
        res1 = np.array(img1)
        assert res1.min() == 0
        assert res1.max() == 255

        # 2. Extremely large negative/positive range [-1000.0, 1000.0]
        arr_large = np.array([[[-1000.0], [0.0], [1000.0]]], dtype=np.float32)
        img_large = SteelDefectDetector._normalize_image_input(arr_large)
        res_large = np.array(img_large)
        assert res_large.shape == (1, 3, 3)
        assert res_large.min() >= 0
        assert res_large.max() <= 255

        # 3. All negative floats [-50.0, -10.0]
        arr_all_neg = np.full((32, 32, 3), -20.0, dtype=np.float32)
        img_all_neg = SteelDefectDetector._normalize_image_input(arr_all_neg)
        res_all_neg = np.array(img_all_neg)
        assert (res_all_neg == 0).all()

        # 4. Tiny positive floats [1e-10, 1e-8]
        arr_tiny = np.full((32, 32, 3), 1e-9, dtype=np.float32)
        img_tiny = SteelDefectDetector._normalize_image_input(arr_tiny)
        res_tiny = np.array(img_tiny)
        assert (res_tiny == 0).all()

    def test_high_dimensional_tensors_fail_cleanly(self):
        """4D and 5D arrays must raise clean ValueError."""
        arr_4d = np.zeros((2, 3, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported numpy array dimensions"):
            SteelDefectDetector._normalize_image_input(arr_4d)

        arr_5d = np.zeros((1, 2, 3, 64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported numpy array dimensions"):
            SteelDefectDetector._normalize_image_input(arr_5d)


# ==============================================================================
# SECTION 3: Out-of-Bounds and Inverted Bounding Boxes in analyze_defect
# ==============================================================================

class TestBoundingBoxStress:
    """Stress-test geometry and metallurgy engine against malformed, inverted, and out-of-bounds bounding boxes."""

    def test_extreme_out_of_bounds_bounding_box(self):
        """Box [-50, -50, 1500, 1500] on (1280, 720) image must clamp cleanly to [0, 0, 1280, 720]."""
        res = analyze_defect(
            defect_class="scratches",
            confidence=0.85,
            box=[-50.0, -50.0, 1500.0, 1500.0],
            image_width=1280,
            image_height=720,
            grade_code="SS_304"
        )
        assert res["bounding_box"] == [0.0, 0.0, 1280.0, 720.0]
        assert res["area_pct"] <= 100.0
        assert res["strip_zone"] == "Strip Center / Body"  # Centroid is at 640.0, strip center

    def test_inverted_coordinates_bounding_box(self):
        """Inverted box [400, 400, 100, 100] must be reordered to [100, 100, 400, 400]."""
        res = analyze_defect(
            defect_class="rolled-in_scale",
            confidence=0.90,
            box=[400.0, 400.0, 100.0, 100.0],
            image_width=1280,
            image_height=720,
            grade_code="SS_316L"
        )
        assert res["bounding_box"] == [100.0, 100.0, 400.0, 400.0]
        assert res["area_pixels"] == 300 * 300
        assert res["aspect_ratio"] == 1.0

    def test_completely_off_screen_negative_box(self):
        """Box [-500, -500, -100, -100] completely outside image frame must clamp to [0, 0, 0, 0] without zero division."""
        res = analyze_defect(
            defect_class="crazing",
            confidence=0.75,
            box=[-500.0, -500.0, -100.0, -100.0],
            image_width=1280,
            image_height=720,
            grade_code="SS_430"
        )
        assert res["bounding_box"] == [0.0, 0.0, 0.0, 0.0]
        assert res["area_pixels"] == 1  # Guarded max(1.0, ...)
        assert res["aspect_ratio"] == 1.0  # Guarded against division by zero

    def test_completely_off_screen_positive_box(self):
        """Box [2000, 2000, 3000, 3000] completely past frame bounds must clamp to [1280, 720, 1280, 720]."""
        res = analyze_defect(
            defect_class="inclusion",
            confidence=0.70,
            box=[2000.0, 2000.0, 3000.0, 3000.0],
            image_width=1280,
            image_height=720,
            grade_code="DUPLEX_2205"
        )
        assert res["bounding_box"] == [1280.0, 720.0, 1280.0, 720.0]
        assert res["area_pixels"] == 1
        assert res["aspect_ratio"] == 1.0

    def test_degenerate_point_box(self):
        """Single point box [250, 350, 250, 350] has zero width and height; must compute area=1 px and no crash."""
        res = analyze_defect(
            defect_class="pitted_surface",
            confidence=0.60,
            box=[250.0, 350.0, 250.0, 350.0],
            image_width=1000,
            image_height=1000,
            grade_code="SS_201"
        )
        assert res["bounding_box"] == [250.0, 350.0, 250.0, 350.0]
        assert res["area_pixels"] == 1
        assert res["aspect_ratio"] == 1.0

    def test_extreme_aspect_ratio_box(self):
        """Ultra-thin slit box [0, 100, 1280, 100.1] spanning entire width."""
        res = analyze_defect(
            defect_class="scratches",
            confidence=0.88,
            box=[0.0, 100.0, 1280.0, 100.1],
            image_width=1280,
            image_height=720,
            grade_code="SS_304"
        )
        assert res["bounding_box"][0] == 0.0
        assert res["bounding_box"][2] == 1280.0
        assert res["strip_zone"] == "Strip Center / Body"  # Centroid at 640.0
        assert res["severity_score"] >= 0.0

    def test_all_five_stainless_grades_with_extreme_boxes(self):
        """Verify behavior across all 5 stainless steel grades with boundary-violating boxes."""
        grades = ["SS_304", "SS_316L", "SS_430", "SS_201", "DUPLEX_2205"]
        for g in grades:
            res = analyze_defect(
                defect_class="rolled-in_scale",
                confidence=0.92,
                box=[-100.0, 100.0, 1500.0, 800.0],
                image_width=1000,
                image_height=500,
                grade_code=g
            )
            assert res["bounding_box"] == [0.0, 100.0, 1000.0, 500.0]
            assert 0.0 <= res["severity_score"] <= 10.0
            assert res["severity_tier"] in ["Negligible", "Low", "Moderate", "High", "Critical"]


# ==============================================================================
# SECTION 4: Line Simulator Latency Profiling & Kinematics Overlap
# ==============================================================================

class TestLineSimulatorStress:
    """Stress-test production line simulator against numerical singularities and latency distributions."""

    def test_profile_latencies_empty_list(self):
        """Empty list must return all zero statistics without ValueError."""
        stats = ProductionLineSimulator.profile_latencies([])
        assert stats["count"] == 0
        assert stats["mean_ms"] == 0.0
        assert stats["median_ms"] == 0.0
        assert stats["p95_ms"] == 0.0
        assert stats["min_ms"] == 0.0
        assert stats["max_ms"] == 0.0

    def test_profile_latencies_single_element_list(self):
        """Single-element list [42.5] must compute percentiles safely without crashing numpy."""
        stats = ProductionLineSimulator.profile_latencies([42.5])
        assert stats["count"] == 1
        assert stats["mean_ms"] == 42.5
        assert stats["median_ms"] == 42.5
        assert stats["std_ms"] == 0.0
        assert stats["min_ms"] == 42.5
        assert stats["max_ms"] == 42.5
        assert stats["p90_ms"] == 42.5
        assert stats["p95_ms"] == 42.5
        assert stats["p99_ms"] == 42.5

    def test_profile_latencies_all_zeros(self):
        """List with 0.0 values."""
        stats = ProductionLineSimulator.profile_latencies([0.0, 0.0, 0.0])
        assert stats["count"] == 3
        assert stats["mean_ms"] == 0.0
        assert stats["max_ms"] == 0.0

    def test_profile_latencies_large_distribution(self):
        """10,000 synthetic latency measurements."""
        data = list(np.random.normal(loc=25.0, scale=3.0, size=10000))
        stats = ProductionLineSimulator.profile_latencies(data)
        assert stats["count"] == 10000
        assert 24.0 < stats["mean_ms"] < 26.0
        assert stats["p95_ms"] > stats["median_ms"]

    def test_kinematics_100_percent_overlap(self):
        """
        100% overlap means effective length per frame = 0.
        Must not cause ZeroDivisionError in `calculate_required_fps` or `compute_line_requirements`.
        """
        sim = ProductionLineSimulator(fov_y_mm=500.0, overlap_pct=100.0)
        assert sim.effective_length_per_frame_m == 0.0

        fps = sim.calculate_required_fps(line_speed_m_per_min=600.0)
        assert np.isfinite(fps)
        assert fps > 0.0

        reqs = sim.compute_line_requirements(line_speed_m_per_min=600.0, measured_latency_ms=20.0)
        assert reqs["required_camera_fps"] > 0.0
        assert reqs["achieved_system_fps"] == 50.0
        assert np.isfinite(reqs["max_sustainable_speed_m_per_min"])
        assert isinstance(reqs["is_realtime_capable"], bool)

    def test_kinematics_over_100_percent_overlap(self):
        """Overlap exceeding 100% (e.g. 150%) results in negative effective length; must be clamped safely."""
        sim = ProductionLineSimulator(fov_y_mm=500.0, overlap_pct=150.0)
        assert sim.effective_length_per_frame_m < 0.0

        fps = sim.calculate_required_fps(line_speed_m_per_min=600.0)
        assert np.isfinite(fps)
        assert fps > 0.0

        reqs = sim.compute_line_requirements(line_speed_m_per_min=600.0, measured_latency_ms=25.0)
        assert reqs["throughput_headroom_ratio"] >= 0.0

    def test_kinematics_zero_line_speed(self):
        """Line speed = 0 m/min (mill stopped). Required FPS = 0. Headroom ratio must not divide by zero."""
        sim = ProductionLineSimulator(fov_y_mm=500.0, overlap_pct=10.0)
        fps = sim.calculate_required_fps(line_speed_m_per_min=0.0)
        assert fps == 0.0

        reqs = sim.compute_line_requirements(line_speed_m_per_min=0.0, measured_latency_ms=25.0)
        assert reqs["target_line_speed_m_per_min"] == 0.0
        assert reqs["required_camera_fps"] == 0.0
        assert np.isfinite(reqs["throughput_headroom_ratio"])
        assert reqs["is_realtime_capable"] is True

    def test_kinematics_zero_latency(self):
        """Zero pipeline latency (measured_latency_ms = 0.0) must not cause division by zero."""
        sim = ProductionLineSimulator()
        reqs = sim.compute_line_requirements(line_speed_m_per_min=600.0, measured_latency_ms=0.0)
        assert np.isfinite(reqs["achieved_system_fps"])
        assert reqs["achieved_system_fps"] == 1000.0  # max(0.001, 0.0) -> 1/0.001


# ==============================================================================
# Standalone Execution Runner
# ==============================================================================

if __name__ == "__main__":
    print("Executing Standalone Empirical Stress Test Suite...")
    pytest.main(["-v", __file__])
