"""
Metadata, Health, Metallurgy Taxonomy, and Stainless Steel Grade Routes.
"""
import os
from pathlib import Path
from fastapi import APIRouter
from backend.config import (
    DEVICE,
    CUDA_DEVICE_NAME,
    DEFECT_CLASSES,
    RAW_IMAGES_DIR
)
from backend.core.grade_profiles import GRADE_PROFILES
from backend.core.metallurgy_engine import EXTENDED_DEFECT_TAXONOMY
from backend.api.schemas import HealthResponse, SampleItem

router = APIRouter(prefix="/api", tags=["Metadata & Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns inspection system operational status and hardware acceleration info."""
    from backend.api.routes_detect import _detector
    yolo_loaded = _detector is not None and _detector.model is not None
    verifier_loaded = _detector is not None and _detector.verifier is not None and _detector.verifier._loaded

    return HealthResponse(
        status="healthy" if yolo_loaded else "initializing",
        system_name="Jindal Stainless AI Defect Inspection Platform",
        device=DEVICE,
        cuda_device_name=CUDA_DEVICE_NAME,
        yolo_model_loaded=yolo_loaded,
        verifier_model_loaded=verifier_loaded,
        defect_classes=DEFECT_CLASSES
    )

@router.get("/grades")
async def get_grades():
    """Retrieve all stainless steel grade tolerance specifications."""
    return {
        "grades": GRADE_PROFILES,
        "default_grade": "SS_304"
    }

@router.get("/taxonomy")
async def get_taxonomy():
    """Retrieve full 25-defect metallurgy taxonomy with root causes and corrective actions."""
    return {
        "taxonomy": EXTENDED_DEFECT_TAXONOMY,
        "total_categories": len(EXTENDED_DEFECT_TAXONOMY)
    }

@router.get("/samples")
async def list_sample_images():
    """List curated real benchmark steel strip images available for instant demo testing."""
    samples = []
    if RAW_IMAGES_DIR.exists():
        for f in sorted(os.listdir(RAW_IMAGES_DIR)):
            if f.endswith((".jpg", ".png", ".jpeg")):
                full_path = RAW_IMAGES_DIR / f
                try:
                    file_size = full_path.stat().st_size
                except OSError:
                    file_size = 0

                # Determine display label
                base_name = f.rsplit(".", 1)[0]
                if "defect_free" in f:
                    label = "Clean Strip (Defect-Free)"
                    category = "Defect-Free Reference"
                elif "steelx" in f:
                    label = f"SteelDefectX: {base_name.replace('steelx_', '').upper()}"
                    category = "SteelDefectX Research Dataset"
                else:
                    parts = base_name.split("_")
                    cls_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
                    label = f"{cls_name.replace('-', ' ').replace('_', ' ').title()} #{parts[-1]}"
                    category = "NEU-DET Benchmark"

                samples.append({
                    "filename": f,
                    "label": label,
                    "category": category,
                    "file_size_bytes": file_size
                })
    return {"samples": samples, "count": len(samples)}
