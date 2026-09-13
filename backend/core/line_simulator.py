"""
Production Line Throughput & Speed Simulator for Jindal Stainless Lines.
Calculates production line speed compatibility, camera acquisition rates,
frame processing latencies, and real-time buffer stability.
"""
import time
import numpy as np
from typing import Dict, Any, List
from backend.config import (
    DEFAULT_LINE_SPEED_M_PER_MIN,
    DEFAULT_CAMERA_FOV_MM,
    MM_PER_PIXEL
)

class ProductionLineSimulator:
    """
    Simulates high-speed continuous strip movement and calculates camera acquisition,
    neural network throughput headroom, and frame drop risk.
    """
    def __init__(
        self,
        fov_x_mm: float = DEFAULT_CAMERA_FOV_MM,      # e.g., 1250 mm strip width
        fov_y_mm: float = 500.0,                      # FOV in strip movement direction
        overlap_pct: float = 10.0                     # 10% overlap to prevent blind spots
    ):
        self.fov_x_mm = fov_x_mm
        self.fov_y_mm = fov_y_mm
        self.overlap_ratio = overlap_pct / 100.0
        self.effective_length_per_frame_m = (self.fov_y_mm * (1.0 - self.overlap_ratio)) / 1000.0

    def calculate_required_fps(self, line_speed_m_per_min: float = DEFAULT_LINE_SPEED_M_PER_MIN) -> float:
        """
        Calculate required camera acquisition rate in FPS to maintain 100% strip surface coverage.
        Guards against division by zero if effective length per frame is non-positive (e.g. overlap >= 100%).
        """
        line_speed_m_per_s = line_speed_m_per_min / 60.0
        eff_len = max(0.001, self.effective_length_per_frame_m)
        return line_speed_m_per_s / eff_len

    def compute_line_requirements(
        self,
        line_speed_m_per_min: float = DEFAULT_LINE_SPEED_M_PER_MIN,
        measured_latency_ms: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculate production line metrics given a target line speed and measured model latency.
        """
        line_speed_m_per_s = line_speed_m_per_min / 60.0
        eff_len = max(0.001, self.effective_length_per_frame_m)
        
        # Required FPS to capture continuous strip without missing any surface
        required_fps = self.calculate_required_fps(line_speed_m_per_min)
        
        # Max FPS achieved by inspection pipeline
        effective_latency_s = max(0.001, measured_latency_ms / 1000.0)
        achieved_fps = 1.0 / effective_latency_s
        
        # Maximum line speed that can be inspected with 100% surface coverage
        max_sustainable_speed_m_per_s = achieved_fps * eff_len
        max_sustainable_speed_m_per_min = max_sustainable_speed_m_per_s * 60.0
        
        # Headroom safety margin (> 1.0 means system is faster than line speed)
        throughput_headroom_ratio = achieved_fps / max(0.01, required_fps)
        is_realtime_capable = throughput_headroom_ratio >= 1.0

        # Frame drop risk
        if throughput_headroom_ratio >= 1.3:
            frame_drop_risk = "Negligible (Safe Buffer > 30%)"
        elif throughput_headroom_ratio >= 1.0:
            frame_drop_risk = "Low (Operating Near Capacity)"
        elif throughput_headroom_ratio >= 0.8:
            frame_drop_risk = "Elevated (Downsampling Recommended)"
        else:
            frame_drop_risk = "Critical (Frame Queue Overflowing)"

        # Linear resolution along strip travel
        pixel_size_mm = MM_PER_PIXEL
        min_detectable_defect_length_mm = round(pixel_size_mm * 5.0, 2)  # ~5 pixels minimum feature

        return {
            "target_line_speed_m_per_min": round(line_speed_m_per_min, 1),
            "target_line_speed_m_per_s": round(line_speed_m_per_s, 2),
            "required_camera_fps": round(required_fps, 1),
            "measured_pipeline_latency_ms": round(measured_latency_ms, 2),
            "achieved_system_fps": round(achieved_fps, 1),
            "max_sustainable_speed_m_per_min": round(max_sustainable_speed_m_per_min, 1),
            "max_sustainable_speed_m_per_s": round(max_sustainable_speed_m_per_s, 2),
            "throughput_headroom_ratio": round(throughput_headroom_ratio, 2),
            "is_realtime_capable": is_realtime_capable,
            "frame_drop_risk": frame_drop_risk,
            "spatial_resolution_mm_per_px": round(pixel_size_mm, 3),
            "min_detectable_defect_size_mm": min_detectable_defect_length_mm,
            "effective_strip_advance_per_frame_m": round(self.effective_length_per_frame_m, 3)
        }

    @staticmethod
    def profile_latencies(
        latencies_ms: List[float]
    ) -> Dict[str, float]:
        """Compute statistical percentiles for benchmark runs. Handles empty input safely."""
        if not latencies_ms or len(latencies_ms) == 0:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        arr = np.array(latencies_ms)
        return {
            "count": int(len(arr)),
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "std_ms": round(float(np.std(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }
