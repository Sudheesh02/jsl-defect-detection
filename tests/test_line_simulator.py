"""
Unit tests for Production Line Speed Simulator and Latency Profiling.
"""
import pytest
from backend.core.line_simulator import ProductionLineSimulator

def test_line_speed_calculations():
    sim = ProductionLineSimulator(fov_x_mm=1250.0, fov_y_mm=500.0, overlap_pct=10.0)
    
    # 500mm FOV with 10% overlap = 0.45m effective advance per frame
    assert round(sim.effective_length_per_frame_m, 2) == 0.45

    # At 600 m/min (10 m/s), required FPS = 10 / 0.45 = 22.2 FPS
    metrics = sim.compute_line_requirements(line_speed_m_per_min=600.0, measured_latency_ms=25.0)
    assert metrics["target_line_speed_m_per_s"] == 10.0
    assert metrics["required_camera_fps"] == 22.2
    assert metrics["achieved_system_fps"] == 40.0 # 1000/25ms
    assert metrics["throughput_headroom_ratio"] == round(40.0 / 22.2, 2)
    assert metrics["is_realtime_capable"] is True
    assert "Safe Buffer" in metrics["frame_drop_risk"]

def test_overburdened_line_speed():
    """Verify that excessive line speed or slow latency flags elevated frame drop risk."""
    sim = ProductionLineSimulator()
    # At 1500 m/min (25 m/s) with 100ms slow latency (10 FPS achieved vs ~55 FPS required)
    metrics = sim.compute_line_requirements(line_speed_m_per_min=1500.0, measured_latency_ms=100.0)
    assert metrics["is_realtime_capable"] is False
    assert metrics["throughput_headroom_ratio"] < 1.0
    assert "Critical" in metrics["frame_drop_risk"]

def test_profile_latencies_statistics():
    sim = ProductionLineSimulator()
    latencies = [20.0, 22.0, 25.0, 28.0, 30.0, 35.0, 50.0]
    stats = sim.profile_latencies(latencies)
    assert stats["count"] == 7
    assert stats["min_ms"] == 20.0
    assert stats["max_ms"] == 50.0
    assert stats["median_ms"] == 28.0
    assert stats["p95_ms"] > stats["median_ms"]
