"""
API Integration tests for FastAPI endpoints.
"""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.config import RAW_IMAGES_DIR

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Jindal Stainless" in data["system_name"]
    assert len(data["defect_classes"]) == 6

def test_grades_endpoint(client):
    response = client.get("/api/grades")
    assert response.status_code == 200
    data = response.json()
    assert "grades" in data
    assert "SS_304" in data["grades"]
    assert "SS_316L" in data["grades"]

def test_taxonomy_endpoint(client):
    response = client.get("/api/taxonomy")
    assert response.status_code == 200
    data = response.json()
    assert "taxonomy" in data
    assert data["total_categories"] == 25
    assert "crazing" in data["taxonomy"]
    assert "roll_printing" in data["taxonomy"]
    assert "blister" in data["taxonomy"]

def test_samples_endpoint(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert data["count"] > 0

def test_sample_detect_endpoint(client):
    response = client.get("/api/sample-detect/scratches_1.jpg?grade_code=SS_304&conf_threshold=0.25")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["defect_count"] > 0
    assert "disposition" in data
    assert "production_line_metrics" in data
    assert data["annotated_image_b64"] is not None

def test_sample_detect_not_found(client):
    response = client.get("/api/sample-detect/imaginary_sample_404.jpg")
    assert response.status_code == 404

def test_sample_detect_path_traversal_forbidden(client):
    """Path traversal attacks must be rejected with 400."""
    response = client.get("/api/sample-detect/../../config.py")
    assert response.status_code in [400, 404]
    response2 = client.get("/api/sample-detect/..%2f..%2fconfig.py")
    assert response2.status_code in [400, 404]

def test_detect_upload_endpoint(client):
    # Create test image in memory
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_strip.jpg", buf, "image/jpeg")}
    response = client.post("/api/detect?grade_code=SS_304&conf_threshold=0.25", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["image_dimensions"]["width"] == 200
    assert data["image_dimensions"]["height"] == 200

def test_detect_upload_invalid_file(client):
    """Uploading corrupt/non-image data should return 400."""
    fake_buf = io.BytesIO(b"this is plainly not an image file content")
    files = {"file": ("corrupt.txt", fake_buf, "text/plain")}
    response = client.post("/api/detect", files=files)
    assert response.status_code == 400

def test_batch_detect_endpoint(client):
    # Upload 2 frames for continuous batch inspection
    files = []
    for i in range(2):
        img = Image.new("RGB", (100, 100), color=(140, 140, 140))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        files.append(("files", (f"frame_{i}.jpg", buf, "image/jpeg")))

    response = client.post("/api/batch-detect?grade_code=SS_304&verify_false_alarms=true", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_frames_inspected"] == 2
    assert "coil_overall_disposition" in data
    assert "frame_summaries" in data
    assert len(data["frame_summaries"]) == 2

def test_benchmark_endpoint(client):
    payload = {
        "iterations": 5,
        "line_speed_m_per_min": 600.0,
        "conf_threshold": 0.25
    }
    response = client.post("/api/benchmark", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["iterations"] == 5
    assert "latency_percentiles_ms" in data
    assert "fps_percentiles" in data
    assert "line_speed_metrics" in data

def test_root_dashboard_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Jindal Stainless" in response.text
    assert "Strip Vision Inspection Viewport" in response.text
