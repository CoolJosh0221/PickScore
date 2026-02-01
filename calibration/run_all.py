#!/usr/bin/env python
"""
Run all calibration analysis phases (1-5).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_phase(phase_num: int, args: list) -> bool:
    """Run a phase script and return success status."""
    script = Path(__file__).parent / f"run_phase{phase_num}.py"
    cmd = [sys.executable, str(script)] + args
    print(f"\n{'='*60}")
    print(f"RUNNING PHASE {phase_num}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all calibration analysis phases")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="calibration_results_v2")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-mc-samples", type=int, default=15)
    parser.add_argument("--pretrained", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--mc-dropout-p", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Baseline Calibration
    phase1_args = [
        "--model-path", args.model_path,
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--n-bins", str(args.n_bins),
        "--batch-size", str(args.batch_size),
        "--num-mc-samples", str(args.num_mc_samples),
        "--pretrained", args.pretrained,
        "--mc-dropout-p", str(args.mc_dropout_p),
    ]
    if not run_phase(1, phase1_args):
        print("Phase 1 failed!")
        return 1

    # Phase 2: Uncertainty-Error Correlation
    phase2_args = [
        "--tensors-path", str(output_dir / "phase1_tensors.pt"),
        "--output-dir", args.output_dir,
    ]
    if not run_phase(2, phase2_args):
        print("Phase 2 failed!")
        return 1

    # Phase 3: Temperature Scaling
    phase3_args = [
        "--model-path", args.model_path,
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--num-mc-samples", str(args.num_mc_samples),
        "--pretrained", args.pretrained,
        "--mc-dropout-p", str(args.mc_dropout_p),
    ]
    if not run_phase(3, phase3_args):
        print("Phase 3 failed!")
        return 1

    # Phase 4: Post-Calibration Evaluation
    phase4_args = [
        "--phase1-tensors", str(output_dir / "phase1_tensors.pt"),
        "--phase3-tensors", str(output_dir / "phase3_tensors.pt"),
        "--output-dir", args.output_dir,
        "--n-bins", str(args.n_bins),
    ]
    if not run_phase(4, phase4_args):
        print("Phase 4 failed!")
        return 1

    # Phase 5: Acquisition Score Comparison
    phase5_args = [
        "--phase1-tensors", str(output_dir / "phase1_tensors.pt"),
        "--phase3-tensors", str(output_dir / "phase3_tensors.pt"),
        "--output-dir", args.output_dir,
        "--top-k", str(args.top_k),
    ]
    if not run_phase(5, phase5_args):
        print("Phase 5 failed!")
        return 1

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
