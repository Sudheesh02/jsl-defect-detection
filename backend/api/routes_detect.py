"""
Defect Detection and Production Benchmark Routes.
"""
import time
import threading
from typing import List, Optional, Union
from pathlib import Path
import torch
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from starlette.concurrency import run_in_threadpool
from PIL import Image
import numpy as np

from backend.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_LINE_SPEED_M_PER_MIN,
    DEFAULT_BATCH_CHUNK_SIZE,
    RAW_IMAGES_DIR
)
from backend.core.detector import SteelDefectDetector
from backend.core.metallurgy_engine import compute_frame_disposition
from backend.api.schemas import (
    DetectionResponse,
    BatchDetectionResponse,
    BenchmarkRequest,
    BenchmarkResponse
)

router = APIRouter(tags=["Defect Inspection & Benchmarks"])

# Global singleton detector instance initialized at app startup
_detector: Optional[SteelDefectDetector] = None
_detector_lock = threading.Lock()

def get_detector() -> SteelDefectDetector:
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = SteelDefectDetector()
    return _detector


@router.post("/detect", response_model=DetectionResponse)
async def detect_defect_on_uploaded_image(
    file: UploadFile = File(...),
    grade_code: str = Query("SS_304", description="Stainless steel grade code (e.g. SS_304, SS_316L, SS_430, SS_201, DUPLEX_2205)"),
    conf_threshold: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.05, le=0.95),
    iou_threshold: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=0.9),
    verify_false_alarms: bool = Query(False, description="Enable secondary ResNet-50 verification for false alarm rejection"),
    line_speed_m_per_min: float = Query(DEFAULT_LINE_SPEED_M_PER_MIN, ge=50.0, le=2500.0)
):
    """
    Perform real-time surface defect detection on an uploaded steel strip image.
    Flags and labels defects, calculates confidence scores, defect severity,
    production line compatibility, and automated coil disposition.
    """
    try:
        try:
            image_bytes = await file.read()
        finally:
            await file.close()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        detector = get_detector()
        
        result = await run_in_threadpool(
            detector.detect,
            image_input=image_bytes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            grade_code=grade_code,
            verify_false_alarms=verify_false_alarms,
            line_speed_m_per_min=line_speed_m_per_min,
            return_annotated_image=True
        )
        return result
    except HTTPException:
        raise
    except (ValueError, IOError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/sample-detect/{filename}", response_model=DetectionResponse)
async def detect_sample_by_filename(
    filename: str,
    grade_code: str = Query("SS_304"),
    conf_threshold: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.05, le=0.95),
    iou_threshold: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=0.9),
    verify_false_alarms: bool = Query(False),
    line_speed_m_per_min: float = Query(DEFAULT_LINE_SPEED_M_PER_MIN)
):
    """
    Run instant 1-click inspection on a real curated benchmark steel image.
    Includes path traversal protection.
    """
    # Security check against path traversal
    try:
        resolved_base = RAW_IMAGES_DIR.resolve()
        file_path = (RAW_IMAGES_DIR / filename).resolve()
        if not file_path.is_relative_to(resolved_base):
            raise HTTPException(status_code=400, detail="Invalid filename parameter: path traversal forbidden.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed filename.")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found.")

    detector = get_detector()
    result = await run_in_threadpool(
        detector.detect,
        image_input=file_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        grade_code=grade_code,
        verify_false_alarms=verify_false_alarms,
        line_speed_m_per_min=line_speed_m_per_min,
        return_annotated_image=True
    )
    return result


@router.post("/batch-detect", response_model=BatchDetectionResponse)
async def batch_inspect_strip_sequence(
    files: List[UploadFile] = File(...),
    grade_code: str = Query("SS_304"),
    conf_threshold: float = Query(DEFAULT_CONF_THRESHOLD, ge=0.05, le=0.95),
    iou_threshold: float = Query(DEFAULT_IOU_THRESHOLD, ge=0.1, le=0.9),
    verify_false_alarms: bool = Query(False, description="Enable secondary ResNet-50 verification for false alarm rejection"),
    line_speed_m_per_min: float = Query(DEFAULT_LINE_SPEED_M_PER_MIN, ge=50.0, le=2500.0)
):
    """
    Simulate continuous coil line inspection across a sequence of consecutive strip frames.
    Uses batched GPU forward passes (chunks of 4) to achieve sustained >58 FPS throughput.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch inspection.")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum limit of 100 frames.")

    # 1. Leak-proof file reading: guaranteed cleanup via try...finally
    raw_images: List[bytes] = []
    filenames: List[str] = []
    try:
        for f in files:
            content = await f.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"File '{f.filename}' is empty (0 bytes).")
            raw_images.append(content)
            filenames.append(f.filename or "unknown.jpg")
    finally:
        for f in files:
            await f.close()

    detector = get_detector()

    # 2. Batched Execution via Threadpool
    try:
        batch_result = await run_in_threadpool(
            detector.detect_batch,
            images=raw_images,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            grade_code=grade_code,
            verify_false_alarms=verify_false_alarms,
            line_speed_m_per_min=line_speed_m_per_min,
            chunk_size=DEFAULT_BATCH_CHUNK_SIZE,
            return_annotated_images=False,
            generate_thumbnails=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batched inference error: {str(e)}")

    # 3. Align actual client filenames into frame summaries
    for idx, (fn, summary) in enumerate(zip(filenames, batch_result["frame_summaries"])):
        summary["filename"] = fn

    return batch_result


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_throughput_benchmark(
    request: BenchmarkRequest
):
    """
    Run stress-test throughput benchmark on the target hardware to measure exact latency percentiles
    (p50, p95, p99), frame rate, and maximum line speed capacity in m/min.
    """
    detector = get_detector()
    latencies = []
    
    # Use real sample image for realistic tensor operations
    sample_file = RAW_IMAGES_DIR / "scratches_1.jpg"
    if sample_file.exists():
        pil_img = Image.open(sample_file).convert("RGB")
    else:
        pil_img = Image.fromarray(np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8))

    # Benchmark loop
    for _ in range(request.iterations):
        res = await run_in_threadpool(
            detector.detect,
            image_input=pil_img,
            conf_threshold=request.conf_threshold,
            line_speed_m_per_min=request.line_speed_m_per_min,
            return_annotated_image=False,
            generate_thumbnails=False
        )
        latencies.append(res["latency"]["total_pipeline_ms"])

    percentiles = detector.line_simulator.profile_latencies(latencies)
    
    # Calculate FPS percentiles
    mean_ms = percentiles.get("mean_ms", 0.0)
    p50_ms = percentiles.get("median_ms", 0.0)
    p95_ms = percentiles.get("p95_ms", 0.0)
    p99_ms = percentiles.get("p99_ms", 0.0)

    fps_percentiles = {
        "mean_fps": round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
        "p50_fps": round(1000.0 / p50_ms, 1) if p50_ms > 0 else 0.0,
        "p95_fps": round(1000.0 / p95_ms, 1) if p95_ms > 0 else 0.0,
        "p99_fps": round(1000.0 / p99_ms, 1) if p99_ms > 0 else 0.0,
    }

    line_metrics = detector.line_simulator.compute_line_requirements(
        line_speed_m_per_min=request.line_speed_m_per_min,
        measured_latency_ms=p95_ms
    )

    return {
        "status": "success",
        "iterations": request.iterations,
        "device": detector.device,
        "latency_percentiles_ms": percentiles,
        "fps_percentiles": fps_percentiles,
        "line_speed_metrics": line_metrics
    }
