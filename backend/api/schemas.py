"""
Pydantic API Schemas for Jindal Stainless Defect Inspection Service.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    system_name: str
    device: str
    cuda_device_name: str
    yolo_model_loaded: bool
    verifier_model_loaded: bool
    defect_classes: List[str]

class DefectItem(BaseModel):
    defect_id: str
    defect_class: str
    full_name: str
    confidence: float
    bounding_box: List[float] = Field(description="[xmin, ymin, xmax, ymax] in pixels")
    area_pixels: int
    area_pct: float
    aspect_ratio: float
    strip_zone: str
    is_edge_defect: bool
    severity_score: float
    severity_tier: str
    is_critical_for_grade: bool
    reworkable: bool
    category: str
    root_cause: str
    metallurgical_hazard: str
    corrective_actions: List[str]
    thumbnail_b64: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    is_potential_false_alarm: bool = False
    consensus_confidence: Optional[float] = None

class DispositionResponse(BaseModel):
    disposition: str
    disposition_code: str
    status_color: str
    overall_severity: float
    quality_grade: str
    quality_score_pct: float
    action_summary: str
    total_defect_count: int
    critical_defect_count: int
    reworkable_defect_count: int

class ProductionLineMetrics(BaseModel):
    target_line_speed_m_per_min: float
    target_line_speed_m_per_s: float
    required_camera_fps: float
    measured_pipeline_latency_ms: float
    achieved_system_fps: float
    max_sustainable_speed_m_per_min: float
    max_sustainable_speed_m_per_s: float
    throughput_headroom_ratio: float
    is_realtime_capable: bool
    frame_drop_risk: str
    spatial_resolution_mm_per_px: float
    min_detectable_defect_size_mm: float
    effective_strip_advance_per_frame_m: float

class LatencyMetrics(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_pipeline_ms: float
    fps: float
    device: str

class DetectionResponse(BaseModel):
    status: str
    image_dimensions: Dict[str, int]
    grade_applied: str
    disposition: DispositionResponse
    defect_count: int
    defects: List[DefectItem]
    production_line_metrics: ProductionLineMetrics
    latency: LatencyMetrics
    annotated_image_b64: Optional[str] = None

class BatchDetectionResponse(BaseModel):
    status: str
    total_frames_inspected: int
    grade_applied: str
    coil_overall_disposition: DispositionResponse
    defect_frequency: Dict[str, int]
    total_defects_found: int
    average_latency_ms: float
    throughput_fps: float
    frame_summaries: List[Dict[str, Any]]

class BenchmarkRequest(BaseModel):
    iterations: int = Field(default=25, ge=5, le=100, description="Number of frames to benchmark")
    line_speed_m_per_min: float = Field(default=600.0, description="Target line speed")
    conf_threshold: float = Field(default=0.25)

class BenchmarkResponse(BaseModel):
    status: str
    iterations: int
    device: str
    latency_percentiles_ms: Dict[str, float]
    fps_percentiles: Dict[str, float]
    line_speed_metrics: Dict[str, Any]

class SampleItem(BaseModel):
    filename: str
    label: str
    category: str
    file_size_bytes: int
