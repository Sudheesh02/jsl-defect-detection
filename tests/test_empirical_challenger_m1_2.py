"""
Adversarial Empirical Stress-Testing Suite for Milestone 1.
Challenger 2: API Endpoints & Metallurgy Logic.

Empirically challenges:
1. /api/batch-detect boundary enforcement (>100 files, 0-byte file, invalid query params)
2. File handle closing and temporary resource leakage verification
3. Metallurgy centroid-based edge proximity for wide body-spanning defects [150, 50, 850, 200]
4. Metallurgy disposition logic for multiple minor critical defects (<4.0 severity)
5. Defect density scaling and disposition escalation in batch mode
"""
import io
import tempfile
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import pytest
import starlette.datastructures
from fastapi.testclient import TestClient

from backend.app import app
from backend.config import DISPOSITION_THRESHOLDS
from backend.core.metallurgy_engine import (
    analyze_defect,
    compute_frame_disposition
)
from backend.core.detector import SteelDefectDetector


client = TestClient(app)


def _create_dummy_image_bytes(width=100, height=100, color=(128, 128, 128)):
    """Helper to generate valid JPEG image bytes in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ==============================================================================
# 1. /api/batch-detect Boundary Enforcement Stress Tests
# ==============================================================================

class TestBatchDetectBoundaryEnforcement:
    """Stress-tests for boundary enforcement on the /api/batch-detect endpoint."""

    def test_batch_exceeding_100_files_rejected_with_400(self):
        """Sending 101 files must immediately return HTTP 400 Bad Request."""
        files = [
            ("files", (f"frame_{i}.jpg", io.BytesIO(b"fake_data"), "image/jpeg"))
            for i in range(101)
        ]
        response = client.post("/api/batch-detect", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "100" in data["detail"]
        assert "maximum limit" in data["detail"].lower() or "limit" in data["detail"].lower()

    def test_batch_large_overload_150_files_rejected_with_400(self):
        """Sending 150 files must also return HTTP 400 without memory exhaustion."""
        files = [
            ("files", (f"frame_{i}.jpg", io.BytesIO(b"fake_data"), "image/jpeg"))
            for i in range(150)
        ]
        response = client.post("/api/batch-detect", files=files)
        assert response.status_code == 400
        assert "maximum limit of 100 frames" in response.json()["detail"]

    def test_batch_with_single_zero_byte_file_rejected_with_400(self):
        """A batch containing a single 0-byte file must return HTTP 400."""
        files = [
            ("files", ("empty_frame.jpg", io.BytesIO(b""), "image/jpeg"))
        ]
        response = client.post("/api/batch-detect", files=files)
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
        assert "0 bytes" in response.json()["detail"]

    def test_batch_with_zero_byte_file_amidst_valid_files_rejected(self):
        """If any file in the batch is 0-byte, endpoint must reject with HTTP 400."""
        valid_img = _create_dummy_image_bytes()
        files = [
            ("files", ("frame_0.jpg", io.BytesIO(valid_img), "image/jpeg")),
            ("files", ("frame_1_corrupt.jpg", io.BytesIO(b""), "image/jpeg")),
            ("files", ("frame_2.jpg", io.BytesIO(valid_img), "image/jpeg")),
        ]
        response = client.post("/api/batch-detect", files=files)
        assert response.status_code == 400
        assert "frame_1_corrupt.jpg" in response.json()["detail"]
        assert "0 bytes" in response.json()["detail"]

    @pytest.mark.parametrize("invalid_conf", [0.01, 0.049, -0.1, 0.96, 1.5, 0.0])
    def test_batch_invalid_conf_threshold_rejected_with_422(self, invalid_conf):
        """conf_threshold must be strictly within [0.05, 0.95], otherwise HTTP 422."""
        files = [("files", ("frame_0.jpg", io.BytesIO(b"valid_placeholder"), "image/jpeg"))]
        response = client.post(f"/api/batch-detect?conf_threshold={invalid_conf}", files=files)
        assert response.status_code == 422

    @pytest.mark.parametrize("invalid_speed", [-10.0, 0.0, 49.9, 2501.0, -100.0])
    def test_batch_invalid_line_speed_rejected_with_422(self, invalid_speed):
        """line_speed_m_per_min must be strictly within [50.0, 2500.0], otherwise HTTP 422."""
        files = [("files", ("frame_0.jpg", io.BytesIO(b"valid_placeholder"), "image/jpeg"))]
        response = client.post(f"/api/batch-detect?line_speed_m_per_min={invalid_speed}", files=files)
        assert response.status_code == 422

    @pytest.mark.parametrize("invalid_iou", [0.05, 0.09, 0.91, 1.2, -0.5])
    def test_batch_invalid_iou_threshold_rejected_with_422(self, invalid_iou):
        """iou_threshold must be strictly within [0.1, 0.9], otherwise HTTP 422."""
        files = [("files", ("frame_0.jpg", io.BytesIO(b"valid_placeholder"), "image/jpeg"))]
        response = client.post(f"/api/batch-detect?iou_threshold={invalid_iou}", files=files)
        assert response.status_code == 422


# ==============================================================================
# 2. File Handle Closing and Resource Leakage Verification
# ==============================================================================

class TestFileHandleClosingVerification:
    """Empirical verification that temporary files and UploadFile handles are properly closed."""

    def test_single_detect_closes_file_handle(self):
        """In /api/detect, the uploaded file handle must be closed after request."""
        raw_bytes = _create_dummy_image_bytes(200, 200)
        
        closed_files = []
        original_close = starlette.datastructures.UploadFile.close

        async def spy_close(self):
            closed_files.append(self.filename)
            return await original_close(self)

        with patch.object(starlette.datastructures.UploadFile, "close", spy_close):
            files = {"file": ("test_leak_check.jpg", io.BytesIO(raw_bytes), "image/jpeg")}
            response = client.post("/api/detect", files=files)
            assert response.status_code == 200
            assert "test_leak_check.jpg" in closed_files

    def test_batch_detect_closes_all_file_handles(self):
        """In /api/batch-detect, all uploaded file handles must be closed."""
        raw_bytes = _create_dummy_image_bytes(100, 100)
        
        closed_files = []
        original_close = starlette.datastructures.UploadFile.close

        async def spy_close(self):
            closed_files.append(self.filename)
            return await original_close(self)

        with patch.object(starlette.datastructures.UploadFile, "close", spy_close):
            files = [
                ("files", (f"batch_leak_{i}.jpg", io.BytesIO(raw_bytes), "image/jpeg"))
                for i in range(3)
            ]
            response = client.post("/api/batch-detect", files=files)
            assert response.status_code == 200
            assert "batch_leak_0.jpg" in closed_files
            assert "batch_leak_1.jpg" in closed_files
            assert "batch_leak_2.jpg" in closed_files

    def test_empty_file_in_detect_closes_handle_on_error(self):
        """When an empty file is passed to /api/detect, close() is still called via finally."""
        closed_files = []
        original_close = starlette.datastructures.UploadFile.close

        async def spy_close(self):
            closed_files.append(self.filename)
            return await original_close(self)

        with patch.object(starlette.datastructures.UploadFile, "close", spy_close):
            files = {"file": ("empty_check.jpg", io.BytesIO(b""), "image/jpeg")}
            response = client.post("/api/detect", files=files)
            assert response.status_code == 400
            assert "empty_check.jpg" in closed_files


# ==============================================================================
# 3. Metallurgy Centroid Edge Proximity Tests
# ==============================================================================

class TestMetallurgyCentroidEdgeProximity:
    """
    Stress-tests for centroid-based edge proximity logic.
    Defects centered in strip body spanning large widths must NOT receive edge penalty.
    """

    def test_wide_defect_centered_in_strip_body_classified_as_center(self):
        """
        User Specification Check:
        Defect centered in strip body with large width [150, 50, 850, 200]
        on a 1000px wide strip has:
        - centroid_x = (150 + 850) / 2.0 = 500.0 px
        - edge_margin_px = 0.12 * 1000 = 120.0 px
        It is in the strip body (500 > 120 and 500 < 880).
        Must be classified as 'Strip Center / Body' with is_edge_defect = False.
        Must NOT receive the 1.35x edge penalty.
        """
        img_w, img_h = 1000, 1000
        wide_box = [150.0, 50.0, 850.0, 200.0]

        res = analyze_defect(
            defect_class="patches",
            confidence=0.80,
            box=wide_box,
            image_width=img_w,
            image_height=img_h,
            grade_code="SS_304"
        )

        assert res["strip_zone"] == "Strip Center / Body"
        assert res["is_edge_defect"] is False
        assert res["bounding_box"] == [150.0, 50.0, 850.0, 200.0]

        # Calculate theoretical severity without edge penalty:
        # For 'patches' in SS_304:
        # base_score = 4.5, grade_weight = 0.8 -> base = 3.6
        # area_px = 700 * 150 = 105,000 px -> area_pct = 10.5%
        # max_acceptable_area = 2.0%
        # area_penalty = min(2.0, (10.5 / 2.0) * 0.5) = 2.0
        # conf_weight = 0.5 + 0.5 * 0.80 = 0.90
        # raw_severity = (3.6 + 2.0) * 1.0 * 0.90 = 5.04
        expected_raw_center = (4.5 * 0.8 + 2.0) * 1.0 * (0.5 + 0.5 * 0.80)
        assert res["severity_score"] == round(expected_raw_center, 2)
        assert res["severity_score"] == 5.04

    def test_wide_defect_scratches_classified_as_center(self):
        """Verify scratches defect spanning [150, 50, 850, 200] is classified as body defect."""
        res = analyze_defect(
            defect_class="scratches",
            confidence=0.85,
            box=[150.0, 50.0, 850.0, 200.0],
            image_width=1000,
            image_height=1000,
            grade_code="SS_304"
        )
        assert res["strip_zone"] == "Strip Center / Body"
        assert res["is_edge_defect"] is False
        assert res["bounding_box"] == [150.0, 50.0, 850.0, 200.0]

    def test_wide_defect_overlapping_both_lateral_margins_classified_as_center(self):
        """
        Adversarial Test: A defect spanning [50, 100, 950, 300] physically crosses
        both the left (x <= 120) and right (x >= 880) edge zones.
        Under flawed naive checks (xmin <= 120 or xmax >= 880), this was flagged as an edge crack.
        Under centroid check, centroid_x = 500.0 px (center), so it must be Strip Center / Body.
        """
        res = analyze_defect(
            defect_class="crease",
            confidence=0.80,
            box=[50.0, 100.0, 950.0, 300.0],
            image_width=1000,
            image_height=1000,
            grade_code="SS_304"
        )
        assert res["strip_zone"] == "Strip Center / Body"
        assert res["is_edge_defect"] is False

    def test_genuine_edge_defect_receives_1_35x_penalty(self):
        """
        A defect with centroid within the 12% margin must be flagged as Strip Edge
        and receive the 1.35x edge risk multiplier.
        """
        img_w, img_h = 1000, 1000
        # Left edge defect: xmin=10, xmax=70 -> centroid=40.0 <= 120.0
        edge_box = [10.0, 50.0, 70.0, 200.0]
        res_edge = analyze_defect(
            defect_class="patches",
            confidence=0.80,
            box=edge_box,
            image_width=img_w,
            image_height=img_h,
            grade_code="SS_304"
        )
        assert res_edge["strip_zone"] == "Strip Edge"
        assert res_edge["is_edge_defect"] is True

        # Equivalent body defect: xmin=470, xmax=530 -> centroid=500.0 (same dimensions 60x150)
        body_box = [470.0, 50.0, 530.0, 200.0]
        res_body = analyze_defect(
            defect_class="patches",
            confidence=0.80,
            box=body_box,
            image_width=img_w,
            image_height=img_h,
            grade_code="SS_304"
        )
        assert res_body["strip_zone"] == "Strip Center / Body"
        assert res_body["is_edge_defect"] is False

        # Edge severity must be strictly higher by ~1.35x
        ratio = res_edge["severity_score"] / res_body["severity_score"]
        assert 1.30 <= ratio <= 1.40

    def test_edge_proximity_boundary_thresholds(self):
        """
        Verify exact boundary behavior around 12% margin (120px on 1000px strip).
        Centroid <= 120.0 -> Edge
        Centroid > 120.0 -> Body
        """
        # Centroid exactly at 120.0 (e.g. [110, 100, 130, 200])
        res_exact = analyze_defect("rolled-in_scale", 0.7, [110.0, 100.0, 130.0, 200.0], 1000, 1000)
        assert res_exact["is_edge_defect"] is True
        assert res_exact["strip_zone"] == "Strip Edge"

        # Centroid just above 120.0 (e.g. [111, 100, 131, 200] -> centroid = 121.0)
        res_just_inside = analyze_defect("rolled-in_scale", 0.7, [111.0, 100.0, 131.0, 200.0], 1000, 1000)
        assert res_just_inside["is_edge_defect"] is False
        assert res_just_inside["strip_zone"] == "Strip Center / Body"


# ==============================================================================
# 4. Metallurgy Disposition Logic (Minor Critical Defects)
# ==============================================================================

class TestMetallurgyDispositionLogic:
    """
    Stress-tests for coil disposition decision tree:
    Two minor inclusions (<4.0 severity) under downgrade threshold yield
    DOWNGRADE TO COMMERCIAL rather than premature SCRAP / REJECT.
    """

    def test_two_minor_critical_inclusions_yield_commercial_downgrade(self):
        """
        User Specification Check:
        Two minor inclusions (< 4.0 severity) with is_critical_for_grade=True
        and composite severity <= 7.5 must yield DOWNGRADE TO COMMERCIAL.
        """
        defects = [
            {"defect_class": "inclusion", "severity_score": 2.2, "is_critical_for_grade": True, "reworkable": False},
            {"defect_class": "inclusion", "severity_score": 2.8, "is_critical_for_grade": True, "reworkable": False}
        ]
        disp = compute_frame_disposition(defects, grade_code="SS_304")
        assert disp["disposition"] == "DOWNGRADE TO COMMERCIAL"
        assert disp["disposition_code"] == "DOWNGRADE"
        assert disp["quality_grade"] == "C"
        assert disp["critical_defect_count"] == 2
        assert "downgrade to commercial" in disp["action_summary"].lower()

    def test_multiple_minor_critical_defects_stay_commercial_if_under_threshold(self):
        """
        Even with 4 minor critical inclusions, if max_severity < 4.0 and composite <= 7.5,
        the coil is downgraded commercially instead of being scrapped.
        """
        defects = [
            {"defect_class": "inclusion", "severity_score": 3.0, "is_critical_for_grade": True, "reworkable": False},
            {"defect_class": "inclusion", "severity_score": 2.5, "is_critical_for_grade": True, "reworkable": False},
            {"defect_class": "inclusion", "severity_score": 3.2, "is_critical_for_grade": True, "reworkable": False},
            {"defect_class": "inclusion", "severity_score": 1.9, "is_critical_for_grade": True, "reworkable": False},
        ]
        disp = compute_frame_disposition(defects, grade_code="SS_304")
        assert disp["disposition_code"] == "DOWNGRADE"
        assert disp["quality_grade"] == "C"

    def test_severe_critical_defects_trigger_scrap(self):
        """
        If multiple critical defects exist and max_severity >= 4.0,
        the decision tree MUST trigger SCRAP / REJECT.
        """
        defects = [
            {"defect_class": "inclusion", "severity_score": 4.5, "is_critical_for_grade": True, "reworkable": False},
            {"defect_class": "inclusion", "severity_score": 2.0, "is_critical_for_grade": True, "reworkable": False}
        ]
        disp = compute_frame_disposition(defects, grade_code="SS_304")
        assert disp["disposition"] == "SCRAP / REJECT"
        assert disp["disposition_code"] == "SCRAP"
        assert disp["quality_grade"] == "F"

    def test_single_severe_critical_defect_under_downgrade_threshold(self):
        """
        A single critical defect (critical_count == 1) with severity <= 7.5
        can still qualify for DOWNGRADE TO COMMERCIAL if composite severity <= 7.5.
        """
        defects = [
            {"defect_class": "crazing", "severity_score": 5.0, "is_critical_for_grade": True, "reworkable": False}
        ]
        disp = compute_frame_disposition(defects, grade_code="SS_304")
        assert disp["disposition_code"] == "DOWNGRADE"
        assert disp["critical_defect_count"] == 1

    def test_composite_severity_exceeding_downgrade_threshold_scraps_regardless(self):
        """
        If composite severity exceeds downgrade_max_severity (7.5),
        the strip MUST be scrapped.
        """
        defects = [
            {"defect_class": "crazing", "severity_score": 8.5, "is_critical_for_grade": True, "reworkable": False},
            {"defect_class": "crazing", "severity_score": 8.0, "is_critical_for_grade": True, "reworkable": False}
        ]
        disp = compute_frame_disposition(defects, grade_code="SS_304")
        assert disp["disposition_code"] == "SCRAP"
        assert disp["quality_grade"] == "F"


# ==============================================================================
# 5. Defect Density in Batch Mode Tests
# ==============================================================================

class TestDefectDensityBatchMode:
    """
    Stress-tests for defect density penalty in continuous batch inspection mode.
    """

    def test_defect_density_penalty_increases_composite_severity(self):
        """
        User Specification Check:
        Batches with high defect density must receive the density penalty:
        density_penalty = min(3.0, (total_defects / total_frames) * 0.3).
        """
        # 10 defects of severity 3.0 across 1 frame (density = 10 defects/frame)
        defects = [
            {"defect_class": "scratches", "severity_score": 3.0, "is_critical_for_grade": False, "reworkable": True}
            for _ in range(10)
        ]
        disp_single = compute_frame_disposition(defects, is_batch=False)
        disp_batch_dense = compute_frame_disposition(defects, is_batch=True, total_frames=1)
        disp_batch_sparse = compute_frame_disposition(defects, is_batch=True, total_frames=10)

        # Non-batch severity: 3.0
        assert disp_single["overall_severity"] == 3.0

        # Dense batch (density = 10 -> penalty = min(3.0, 10 * 0.3) = 3.0 -> severity = 6.0)
        assert disp_batch_dense["overall_severity"] == 6.0

        # Sparse batch (density = 1 -> penalty = min(3.0, 1 * 0.3) = 0.3 -> severity = 3.3)
        assert disp_batch_sparse["overall_severity"] == 3.3

        assert disp_batch_dense["overall_severity"] > disp_batch_sparse["overall_severity"] > disp_single["overall_severity"]

    def test_defect_density_penalty_cap_at_3_0(self):
        """Density penalty must cap cleanly at 3.0 regardless of how massive the defect count is."""
        # 100 defects in 1 frame -> density = 100 -> 100 * 0.3 = 30.0 -> capped at 3.0
        massive_defects = [
            {"defect_class": "patches", "severity_score": 2.0, "is_critical_for_grade": False, "reworkable": True}
            for _ in range(100)
        ]
        disp = compute_frame_disposition(massive_defects, is_batch=True, total_frames=1)
        # Base composite = 2.0, + capped penalty 3.0 = 5.0
        assert disp["overall_severity"] == 5.0

    def test_density_penalty_escalates_disposition_tier(self):
        """
        A cluster of reworkable defects that would be REWORK in a single frame
        escalates to DOWNGRADE TO COMMERCIAL when concentrated densely in a batch.
        """
        # Severity 2.5 normally yields REWORK (since 2.5 <= 4.5 and all reworkable)
        reworkable_defects = [
            {"defect_class": "patches", "severity_score": 2.5, "is_critical_for_grade": False, "reworkable": True}
            for _ in range(10)
        ]
        disp_single = compute_frame_disposition(reworkable_defects, is_batch=False)
        assert disp_single["disposition_code"] == "REWORK"

        # In dense batch (total_frames=1), density penalty is +3.0 -> severity becomes 5.5
        # 5.5 exceeds rework_max_severity (4.5), so it escalates to DOWNGRADE
        disp_batch = compute_frame_disposition(reworkable_defects, is_batch=True, total_frames=1)
        assert disp_batch["disposition_code"] == "DOWNGRADE"
        assert disp_batch["quality_grade"] == "C"

    def test_batch_detect_endpoint_returns_coherent_batch_disposition(self):
        """
        Integration test verifying that /api/batch-detect calculates and returns
        coil_overall_disposition incorporating density penalties across real uploads.
        """
        # Create 2 dummy images
        img_bytes = _create_dummy_image_bytes(100, 100)
        files = [
            ("files", ("batch_f1.jpg", io.BytesIO(img_bytes), "image/jpeg")),
            ("files", ("batch_f2.jpg", io.BytesIO(img_bytes), "image/jpeg")),
        ]
        response = client.post("/api/batch-detect?grade_code=SS_304", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "coil_overall_disposition" in data
        disp = data["coil_overall_disposition"]
        assert "disposition_code" in disp
        assert "overall_severity" in disp
        assert "quality_grade" in disp
        assert data["total_frames_inspected"] == 2
