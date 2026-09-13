"""
Command-line Production Line Throughput and False Alarm Calibration Tool.
Profiles single-frame and batched inference throughput on NVIDIA RTX GPU and CPU,
verifies >25 FPS GPU throughput and >600 m/min line speed compatibility,
and audits 0.0% False Alarm Rate (FAR) across diverse stainless steel surface finishes:
Mirror BA, Brushed No. 4, and 2B Matte.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from PIL import Image

from backend.config import RAW_IMAGES_DIR, DEFAULT_LINE_SPEED_M_PER_MIN
from backend.core.detector import SteelDefectDetector

# Surface finish definitions for clean strip false alarm rejection
CLEAN_FINISH_DEFINITIONS: List[Tuple[str, str, str]] = [
    ("Standard Clean Strip 1", "defect_free_strip_1.jpg", "Baseline Clean Cold Rolled"),
    ("Standard Clean Strip 2", "defect_free_strip_2.jpg", "Baseline Clean Annealed"),
    ("Mirror BA", "defect_free_strip_3_mirror.jpg", "Bright Annealed High-Gloss"),
    ("Brushed No. 4", "defect_free_strip_4_brushed.jpg", "Directional Abrasive Grain"),
    ("2B Matte", "defect_free_strip_5_mill2b.jpg", "Cold Rolled Pickled Skin-Pass"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jindal Stainless AI Defect Detection - Production Line Throughput & FAR Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of iterations per benchmark run",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4",
        help="Comma-separated batch sizes to profile (e.g. '1,2,4' or '1,2,4,8')",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for defect detection",
    )
    parser.add_argument(
        "--line-speed",
        type=float,
        default=DEFAULT_LINE_SPEED_M_PER_MIN,
        help="Target mill line speed in m/min (default: 600.0 m/min)",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations to prime GPU boost clock (P0 state)",
    )
    parser.add_argument(
        "--enable-tf32",
        action="store_true",
        default=True,
        help="Enable TensorFloat-32 (TF32) for fast matrix multiplication on Ampere/Ada/Blackwell GPUs",
    )
    return parser.parse_args()


def benchmark_batch_throughput(
    detector: SteelDefectDetector,
    sample_img: Image.Image,
    batch_size: int,
    iterations: int,
    conf_threshold: float,
    line_speed_m_per_min: float,
) -> Dict[str, Any]:
    """Profile neural and pipeline latency for a given batch size."""
    batch_images = [sample_img] * batch_size
    batch_latencies = []

    # Warmup for this specific batch shape
    for _ in range(3):
        detector.model(batch_images, conf=conf_threshold, verbose=False, device=detector.device)
    if detector.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    for _ in range(iterations):
        t0 = time.perf_counter()
        results = detector.model(
            batch_images,
            conf=conf_threshold,
            verbose=False,
            device=detector.device,
        )
        if detector.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000.0)

    stats = detector.line_simulator.profile_latencies(batch_latencies)
    mean_batch_ms = stats["mean_ms"]
    p95_batch_ms = stats["p95_ms"]

    # Per-frame effective latency
    mean_per_frame_ms = mean_batch_ms / batch_size
    p95_per_frame_ms = p95_batch_ms / batch_size
    effective_fps = 1000.0 / max(0.1, mean_per_frame_ms)

    line_req = detector.line_simulator.compute_line_requirements(
        line_speed_m_per_min=line_speed_m_per_min,
        measured_latency_ms=p95_per_frame_ms,
    )

    return {
        "batch_size": batch_size,
        "mean_batch_ms": mean_batch_ms,
        "p95_batch_ms": p95_batch_ms,
        "mean_per_frame_ms": mean_per_frame_ms,
        "p95_per_frame_ms": p95_per_frame_ms,
        "effective_fps": effective_fps,
        "max_line_speed_m_per_min": line_req["max_sustainable_speed_m_per_min"],
        "max_line_speed_m_per_s": line_req["max_sustainable_speed_m_per_s"],
        "headroom_ratio": line_req["throughput_headroom_ratio"],
        "frame_drop_risk": line_req["frame_drop_risk"],
        "required_camera_fps": line_req["required_camera_fps"],
    }


def audit_clean_strip_finishes(
    detector: SteelDefectDetector,
    conf_threshold: float = 0.25,
) -> Tuple[List[Dict[str, str]], float]:
    """
    Validate false alarm rejection across diverse stainless steel surface finishes:
    Mirror BA, Brushed No. 4, 2B Matte, and standard cold-rolled strip.
    """
    finish_results = []
    total_clean = 0
    total_false_alarms = 0

    for finish_name, file_name, desc in CLEAN_FINISH_DEFINITIONS:
        img_path = RAW_IMAGES_DIR / file_name
        if not img_path.exists():
            finish_results.append({
                "finish_name": finish_name,
                "file_name": file_name,
                "description": desc,
                "status": "FILE MISSING",
                "defects_raw": "N/A",
                "defects_verified": "N/A",
                "disposition": "ERROR",
                "far": "N/A",
            })
            continue

        clean_img = Image.open(img_path).convert("RGB")
        total_clean += 1

        # Test raw detector
        res_raw = detector.detect(
            clean_img,
            conf_threshold=conf_threshold,
            verify_false_alarms=False,
            return_annotated_image=False,
            generate_thumbnails=False,
        )
        # Test with ResNet-50 false-alarm verifier
        res_ver = detector.detect(
            clean_img,
            conf_threshold=conf_threshold,
            verify_false_alarms=True,
            return_annotated_image=False,
            generate_thumbnails=False,
        )

        defects_raw = res_raw["defect_count"]
        # Active confirmed defects after verifier
        active_defects = [
            d for d in res_ver["defects"]
            if not d.get("is_potential_false_alarm", False)
        ]
        defects_ver = len(active_defects)
        is_fa = defects_ver > 0
        if is_fa:
            total_false_alarms += 1

        finish_results.append({
            "finish_name": finish_name,
            "file_name": file_name,
            "description": desc,
            "status": "PASS" if not is_fa else "FAIL",
            "defects_raw": str(defects_raw),
            "defects_verified": str(defects_ver),
            "disposition": res_ver["disposition"]["disposition_code"],
            "far": "0.0%" if not is_fa else "100.0%",
        })

    overall_far = (total_false_alarms / max(1, total_clean)) * 100.0
    return finish_results, overall_far


def run_benchmark():
    args = parse_args()

    # Configure Tensor Core TF32 acceleration
    if args.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 85)
    print("  JINDAL STAINLESS PRODUCTION LINE BENCHMARK & HARDWARE CALIBRATION")
    print("=" * 85)

    detector = SteelDefectDetector()
    sample_file = RAW_IMAGES_DIR / "scratches_1.jpg"
    assert sample_file.exists(), f"Sample image {sample_file} not found."
    sample_img = Image.open(sample_file).convert("RGB")

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {detector.device.upper()} | Name: {dev_name}")
    print(f"Target Line Speed: {args.line_speed:.1f} m/min | Iterations: {args.iterations} | Warmup: {args.warmup_iterations}")

    # Extended GPU Warmup to lock P0 power state
    if detector.device == "cuda" and torch.cuda.is_available():
        print(f"\n[0] Executing {args.warmup_iterations} Warmup Iterations (Locking GPU P0 Boost State)...")
        for _ in range(args.warmup_iterations):
            detector.model(sample_img, verbose=False, device="cuda")
        torch.cuda.synchronize()
        print("    GPU Boost Clocks Active & CUDA Kernel Caches Initialized.")

    # Parse batch sizes
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip().isdigit()]

    # Section 1: Batched Throughput & Latency Matrix
    print(f"\n[1] THROUGHPUT PROFILING ACROSS BATCH SIZES ({args.batch_sizes}):")
    print(f"{'Batch':<7} | {'Mean Latency':<16} | {'p95 Latency':<15} | {'FPS':<10} | {'Max Line Speed':<16} | {'Headroom':<10} | {'Status':<12}")
    print("-" * 96)

    perf_records = []
    for bs in batch_sizes:
        rec = benchmark_batch_throughput(
            detector=detector,
            sample_img=sample_img,
            batch_size=bs,
            iterations=args.iterations,
            conf_threshold=args.conf_threshold,
            line_speed_m_per_min=args.line_speed,
        )
        perf_records.append(rec)
        lat_str = f"{rec['mean_per_frame_ms']:.2f} ms/frame"
        p95_str = f"{rec['p95_per_frame_ms']:.2f} ms/frame"
        speed_str = f"{rec['max_line_speed_m_per_min']:.1f} m/min"
        headroom_str = f"{rec['headroom_ratio']:.2f}x"
        status_str = "PASS (>25 FPS)" if rec['effective_fps'] >= 25.0 else "SUB-OPTIMAL"
        print(f"B={rec['batch_size']:<5} | {lat_str:<16} | {p95_str:<15} | {rec['effective_fps']:>6.1f} FPS | {speed_str:<16} | {headroom_str:<10} | {status_str:<12}")
    print("-" * 96)

    # Section 2: Mill Speed Compatibility Analysis
    best_perf = max(perf_records, key=lambda x: x["effective_fps"])
    print(f"\n[2] MILL SPEED COMPATIBILITY SUMMARY ({args.line_speed:.1f} m/min Cold Finishing):")
    print(f"  Required Camera Acquisition: {best_perf['required_camera_fps']:.1f} FPS")
    print(f"  Sequential (Batch 1) FPS:    {perf_records[0]['effective_fps']:.1f} FPS (Max Line Speed: {perf_records[0]['max_line_speed_m_per_min']:.1f} m/min)")
    print(f"  Optimal Batched FPS:         {best_perf['effective_fps']:.1f} FPS (at Batch Size {best_perf['batch_size']})")
    print(f"  Max Sustainable Line Speed:  {best_perf['max_line_speed_m_per_min']:.1f} m/min ({best_perf['max_line_speed_m_per_s']:.2f} m/s)")
    print(f"  Throughput Safety Headroom:  {best_perf['headroom_ratio']:.2f}x ({best_perf['frame_drop_risk']})")

    fps_pass = best_perf['effective_fps'] >= 25.0
    speed_pass = best_perf['max_line_speed_m_per_min'] >= args.line_speed
    print(f"  Acceptance Check (>25 FPS):   {'[PASS]' if fps_pass else '[FAIL]'} ({best_perf['effective_fps']:.1f} FPS achieved)")
    print(f"  Acceptance Check (>600 m/min): {'[PASS]' if speed_pass else '[FAIL]'} ({best_perf['max_line_speed_m_per_min']:.1f} m/min achieved)")

    # Section 3: Multi-Finish Clean Strip False Alarm Rejection Test
    print(f"\n[3] MULTI-FINISH CLEAN STRIP FALSE ALARM REJECTION (FAR) AUDIT:")
    finish_results, overall_far = audit_clean_strip_finishes(detector, conf_threshold=args.conf_threshold)
    print(f"{'Finish Name':<24} | {'File Name':<34} | {'Raw Defects':<12} | {'Verified':<10} | {'Disposition':<12} | {'FAR'}")
    print("-" * 105)
    for fr in finish_results:
        print(f"{fr['finish_name']:<24} | {fr['file_name']:<34} | {fr['defects_raw']:<12} | {fr['defects_verified']:<10} | {fr['disposition']:<12} | {fr['far']}")
    print("-" * 105)
    print(f"  Total Clean Frames Tested:    {len([f for f in finish_results if f['status'] != 'FILE MISSING'])}")
    print(f"  Overall False Alarm Rate:     {overall_far:.1f}%")
    print(f"  Acceptance Check (0.0% FAR):  {'[PASS]' if overall_far == 0.0 else '[FAIL]'}")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark()
