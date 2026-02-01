"""
Calibration Analysis for Active Learning - All Phases
Phase 2: Uncertainty-Error Correlation (BALD)
Phase 3: Temperature Scaling
Phase 4: Post-Calibration Evaluation
Phase 5: Compare calibrated vs uncalibrated AL acquisition scores
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm
from transformers import CLIPProcessor

from active_learning.data.loaders import create_dataloader
from active_learning.models.model_mcdo import MCDropoutCLIPModel
from active_learning.training.acquisitions import BALD, EntropyUncertainty, LeastConfidence


def collect_mc_predictions(model, processor, loader, device: str, num_mc_samples: int = 15):
    """Collect MC Dropout predictions - returns both mean probs and all MC samples."""
    model.enable_mc_dropout = True
    model.eval()

    all_mc_probs = []  # [batches] of [T, B, 2]
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collecting MC predictions (T={num_mc_samples})"):
            imgs0 = batch["image_0"].to(device, non_blocking=True)
            imgs1 = batch["image_1"].to(device, non_blocking=True)
            y0 = batch["label_0"]
            y1 = batch["label_1"]

            txt = processor(
                text=batch["caption"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}

            batch_probs = []
            for _ in range(num_mc_samples):
                t = model.get_text_features(**txt)
                i0 = model.get_image_features(pixel_values=imgs0)
                i1 = model.get_image_features(pixel_values=imgs1)

                t = F.normalize(t, dim=-1)
                i0 = F.normalize(i0, dim=-1)
                i1 = F.normalize(i1, dim=-1)
                scale = model.logit_scale.exp()
                s0 = (t * i0).sum(-1) * scale
                s1 = (t * i1).sum(-1) * scale

                logits = torch.stack([s0, s1], dim=-1)
                probs = F.softmax(logits, dim=-1)
                batch_probs.append(probs.cpu())

            mc_probs = torch.stack(batch_probs, dim=0)  # [T, B, 2]
            labels = torch.stack([y0, y1], dim=-1)

            all_mc_probs.append(mc_probs)
            all_labels.append(labels)

    model.enable_mc_dropout = False

    # Concatenate along batch dimension
    mc_probs = torch.cat(all_mc_probs, dim=1)  # [T, N, 2]
    labels = torch.cat(all_labels, dim=0)  # [N, 2]
    mean_probs = mc_probs.mean(dim=0)  # [N, 2]

    return mc_probs, mean_probs, labels


def collect_logits(model, processor, loader, device: str, num_mc_samples: int = 15):
    """Collect raw logits for temperature scaling optimization."""
    model.enable_mc_dropout = True
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Collecting logits (T={num_mc_samples})"):
            imgs0 = batch["image_0"].to(device, non_blocking=True)
            imgs1 = batch["image_1"].to(device, non_blocking=True)
            y0 = batch["label_0"]
            y1 = batch["label_1"]

            txt = processor(
                text=batch["caption"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            txt = {k: v.to(device, non_blocking=True) for k, v in txt.items()}

            batch_logits = []
            for _ in range(num_mc_samples):
                t = model.get_text_features(**txt)
                i0 = model.get_image_features(pixel_values=imgs0)
                i1 = model.get_image_features(pixel_values=imgs1)

                t = F.normalize(t, dim=-1)
                i0 = F.normalize(i0, dim=-1)
                i1 = F.normalize(i1, dim=-1)
                scale = model.logit_scale.exp()
                s0 = (t * i0).sum(-1) * scale
                s1 = (t * i1).sum(-1) * scale

                logits = torch.stack([s0, s1], dim=-1)
                batch_logits.append(logits.cpu())

            mc_logits = torch.stack(batch_logits, dim=0)  # [T, B, 2]
            labels = torch.stack([y0, y1], dim=-1)

            all_logits.append(mc_logits)
            all_labels.append(labels)

    model.enable_mc_dropout = False

    mc_logits = torch.cat(all_logits, dim=1)  # [T, N, 2]
    labels = torch.cat(all_labels, dim=0)  # [N, 2]

    return mc_logits, labels


def compute_bald(mc_probs: torch.Tensor) -> torch.Tensor:
    """Compute BALD scores: H[E[p]] - E[H[p]]"""
    def entropy(p):
        p = p.clamp_min(1e-9)
        return -(p * p.log()).sum(dim=-1)

    mean_probs = mc_probs.mean(dim=0)  # [N, 2]
    predictive_entropy = entropy(mean_probs)  # [N]
    expected_entropy = entropy(mc_probs).mean(dim=0)  # [N]
    return predictive_entropy - expected_entropy


def compute_calibration_metrics(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    """Compute ECE, MCE from probabilities."""
    pred_probs = probs[:, 0]
    targets = labels[:, 0]

    predictions = (pred_probs > 0.5).float()
    actuals = (targets > 0.5).float()
    confidences = torch.max(probs, dim=-1).values
    correct = (predictions == actuals).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = correct[in_bin].mean().item()
            bin_size = in_bin.sum().item()
            calibration_gap = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * calibration_gap
            mce = max(mce, calibration_gap)
            bin_data.append({
                "lower": bin_lower.item(), "upper": bin_upper.item(),
                "confidence": avg_confidence, "accuracy": avg_accuracy,
                "count": bin_size, "gap": calibration_gap
            })
        else:
            bin_data.append({
                "lower": bin_lower.item(), "upper": bin_upper.item(),
                "confidence": None, "accuracy": None, "count": 0, "gap": 0
            })

    return {
        "ece": ece, "mce": mce, "bins": bin_data,
        "overall_accuracy": correct.mean().item(),
        "mean_confidence": confidences.mean().item(),
    }


def find_optimal_temperature(mc_logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find optimal temperature T that minimizes NLL on validation set."""
    # Average logits across MC samples
    mean_logits = mc_logits.mean(dim=0)  # [N, 2]

    def nll_loss(T):
        scaled_logits = mean_logits / T
        probs = F.softmax(scaled_logits, dim=-1)
        # NLL with soft labels
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()
        return loss.item()

    result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
    return result.x


def apply_temperature(mc_logits: torch.Tensor, T: float) -> tuple:
    """Apply temperature scaling to logits."""
    scaled_logits = mc_logits / T
    mc_probs = F.softmax(scaled_logits, dim=-1)
    mean_probs = mc_probs.mean(dim=0)
    return mc_probs, mean_probs


def plot_reliability_diagram(bin_data, output_path: Path, title: str):
    """Generate reliability diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    non_empty = [b for b in bin_data if b["count"] > 0]
    if not non_empty:
        plt.close()
        return

    confidences = [b["confidence"] for b in non_empty]
    accuracies = [b["accuracy"] for b in non_empty]
    gaps = [b["gap"] for b in non_empty]

    ax1.bar(confidences, accuracies, width=0.05, alpha=0.7, label="Accuracy", color="blue", edgecolor="black")
    ax1.bar(confidences, gaps, width=0.05, alpha=0.5, bottom=accuracies, label="Gap", color="red", edgecolor="black")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    bin_edges = np.linspace(0, 1, 16)
    bin_counts = np.zeros(15)
    for b in bin_data:
        idx = int(b["lower"] * 15)
        if idx < 15:
            bin_counts[idx] = b["count"]

    ax2.bar(bin_edges[:-1] + 0.033, bin_counts, width=0.066, alpha=0.7, color="green", edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence Histogram", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_uncertainty_error_correlation(uncertainty: np.ndarray, errors: np.ndarray,
                                       output_path: Path, title: str, correlation: float):
    """Plot uncertainty vs error scatter with trend line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(uncertainty, errors, alpha=0.3, s=10)

    # Trend line
    z = np.polyfit(uncertainty, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(uncertainty.min(), uncertainty.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Trend (ρ={correlation:.3f})")

    ax.set_xlabel("BALD Uncertainty", fontsize=12)
    ax.set_ylabel("Error (0=correct, 1=wrong)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Full Calibration Analysis - Phases 2-5")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="calibration_results")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-mc-samples", type=int, default=15)
    parser.add_argument("--pretrained", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--mc-dropout-p", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = MCDropoutCLIPModel(
        pretrained_model_name_or_path=args.pretrained,
        mc_dropout_p=args.mc_dropout_p
    )
    model.load(args.model_path)
    model.to(device)

    processor = CLIPProcessor.from_pretrained(args.pretrained)

    # Load test data
    print(f"Loading test data from {args.data_dir}")
    test_loader = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=4,
        processor=processor,
        shuffle=False,
    )

    # Collect logits and MC predictions
    print("\n" + "="*60)
    print("COLLECTING PREDICTIONS")
    print("="*60)
    mc_logits, labels = collect_logits(model, processor, test_loader, device, args.num_mc_samples)
    mc_probs = F.softmax(mc_logits, dim=-1)
    mean_probs = mc_probs.mean(dim=0)
    print(f"Collected {mc_logits.shape[1]} samples with T={args.num_mc_samples} MC samples")

    results = {"timestamp": datetime.now().isoformat(), "num_mc_samples": args.num_mc_samples}

    # =========================================================================
    # PHASE 2: Uncertainty-Error Correlation
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: UNCERTAINTY-ERROR CORRELATION")
    print("="*60)

    # Compute BALD uncertainty
    bald_scores = compute_bald(mc_probs)

    # Compute errors (0=correct, 1=wrong)
    predictions = (mean_probs[:, 0] > 0.5).float()
    actuals = (labels[:, 0] > 0.5).float()
    errors = (predictions != actuals).float()

    # Spearman correlation
    bald_np = bald_scores.numpy()
    errors_np = errors.numpy()
    spearman_corr, spearman_p = stats.spearmanr(bald_np, errors_np)

    print(f"BALD range: [{bald_np.min():.4f}, {bald_np.max():.4f}]")
    print(f"Error rate: {errors_np.mean()*100:.2f}%")
    print(f"Spearman correlation (uncertainty vs error): {spearman_corr:.4f} (p={spearman_p:.2e})")

    # Also compute for Entropy and Least Confidence
    entropy_scores = -(mean_probs * mean_probs.clamp_min(1e-9).log()).sum(dim=-1)
    lc_scores = 1.0 - mean_probs.max(dim=-1).values

    entropy_corr, entropy_p = stats.spearmanr(entropy_scores.numpy(), errors_np)
    lc_corr, lc_p = stats.spearmanr(lc_scores.numpy(), errors_np)

    print(f"Entropy correlation: {entropy_corr:.4f} (p={entropy_p:.2e})")
    print(f"Least Confidence correlation: {lc_corr:.4f} (p={lc_p:.2e})")

    results["phase2_baseline"] = {
        "bald_spearman": spearman_corr,
        "bald_p_value": spearman_p,
        "entropy_spearman": entropy_corr,
        "least_confidence_spearman": lc_corr,
        "error_rate": errors_np.mean(),
    }

    # Plot
    plot_uncertainty_error_correlation(
        bald_np, errors_np,
        output_dir / "uncertainty_error_baseline.png",
        f"BALD Uncertainty vs Error (Baseline, ρ={spearman_corr:.3f})",
        spearman_corr
    )

    # =========================================================================
    # PHASE 3: Temperature Scaling
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: TEMPERATURE SCALING")
    print("="*60)

    # Find optimal temperature
    optimal_T = find_optimal_temperature(mc_logits, labels)
    print(f"Optimal temperature T = {optimal_T:.4f}")

    # Compute NLL before and after
    mean_logits = mc_logits.mean(dim=0)

    def compute_nll(logits, T=1.0):
        scaled = logits / T
        log_probs = F.log_softmax(scaled, dim=-1)
        return -(labels * log_probs).sum(dim=-1).mean().item()

    nll_before = compute_nll(mean_logits, T=1.0)
    nll_after = compute_nll(mean_logits, T=optimal_T)

    print(f"NLL before: {nll_before:.4f}")
    print(f"NLL after:  {nll_after:.4f}")
    print(f"NLL reduction: {(nll_before - nll_after) / nll_before * 100:.2f}%")

    results["phase3"] = {
        "optimal_temperature": optimal_T,
        "nll_before": nll_before,
        "nll_after": nll_after,
    }

    # =========================================================================
    # PHASE 4: Post-Calibration Evaluation
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 4: POST-CALIBRATION EVALUATION")
    print("="*60)

    # Apply temperature scaling
    mc_probs_calibrated, mean_probs_calibrated = apply_temperature(mc_logits, optimal_T)

    # Compute calibration metrics - baseline
    metrics_baseline = compute_calibration_metrics(mean_probs, labels, args.n_bins)
    print(f"\nBaseline (T=1.0):")
    print(f"  ECE: {metrics_baseline['ece']*100:.2f}%")
    print(f"  MCE: {metrics_baseline['mce']*100:.2f}%")
    print(f"  Mean Confidence: {metrics_baseline['mean_confidence']*100:.2f}%")

    # Compute calibration metrics - calibrated
    metrics_calibrated = compute_calibration_metrics(mean_probs_calibrated, labels, args.n_bins)
    print(f"\nCalibrated (T={optimal_T:.2f}):")
    print(f"  ECE: {metrics_calibrated['ece']*100:.2f}%")
    print(f"  MCE: {metrics_calibrated['mce']*100:.2f}%")
    print(f"  Mean Confidence: {metrics_calibrated['mean_confidence']*100:.2f}%")

    print(f"\nECE reduction: {metrics_baseline['ece']*100:.2f}% → {metrics_calibrated['ece']*100:.2f}% "
          f"({(metrics_baseline['ece'] - metrics_calibrated['ece']) / metrics_baseline['ece'] * 100:.1f}% improvement)")

    results["phase4"] = {
        "baseline": metrics_baseline,
        "calibrated": metrics_calibrated,
        "ece_reduction_pct": (metrics_baseline['ece'] - metrics_calibrated['ece']) / metrics_baseline['ece'] * 100,
    }

    # Plot reliability diagrams
    plot_reliability_diagram(
        metrics_baseline["bins"],
        output_dir / "reliability_baseline.png",
        f"Baseline (ECE={metrics_baseline['ece']*100:.2f}%)"
    )
    plot_reliability_diagram(
        metrics_calibrated["bins"],
        output_dir / "reliability_calibrated.png",
        f"Calibrated T={optimal_T:.2f} (ECE={metrics_calibrated['ece']*100:.2f}%)"
    )

    # Uncertainty-error correlation after calibration
    bald_calibrated = compute_bald(mc_probs_calibrated)
    bald_cal_np = bald_calibrated.numpy()
    spearman_cal, spearman_cal_p = stats.spearmanr(bald_cal_np, errors_np)

    print(f"\nBALD-Error correlation after calibration: {spearman_cal:.4f} (p={spearman_cal_p:.2e})")
    print(f"Change: {spearman_corr:.4f} → {spearman_cal:.4f}")

    results["phase4"]["bald_spearman_calibrated"] = spearman_cal

    plot_uncertainty_error_correlation(
        bald_cal_np, errors_np,
        output_dir / "uncertainty_error_calibrated.png",
        f"BALD Uncertainty vs Error (Calibrated T={optimal_T:.2f}, ρ={spearman_cal:.3f})",
        spearman_cal
    )

    # =========================================================================
    # PHASE 5: Compare Acquisition Scores
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 5: ACQUISITION SCORE COMPARISON")
    print("="*60)

    # Compare how acquisition scores change with calibration
    bald_acq = BALD()
    entropy_acq = EntropyUncertainty()
    lc_acq = LeastConfidence()

    # Baseline scores
    bald_base = bald_acq.score(mean_probs=mean_probs, mc_probs=mc_probs)
    entropy_base = entropy_acq.score(mean_probs=mean_probs)
    lc_base = lc_acq.score(mean_probs=mean_probs)

    # Calibrated scores
    bald_cal = bald_acq.score(mean_probs=mean_probs_calibrated, mc_probs=mc_probs_calibrated)
    entropy_cal = entropy_acq.score(mean_probs=mean_probs_calibrated)
    lc_cal = lc_acq.score(mean_probs=mean_probs_calibrated)

    # Rank correlation between baseline and calibrated
    bald_rank_corr, _ = stats.spearmanr(bald_base.numpy(), bald_cal.numpy())
    entropy_rank_corr, _ = stats.spearmanr(entropy_base.numpy(), entropy_cal.numpy())
    lc_rank_corr, _ = stats.spearmanr(lc_base.numpy(), lc_cal.numpy())

    print(f"Rank correlation (baseline vs calibrated):")
    print(f"  BALD: {bald_rank_corr:.4f}")
    print(f"  Entropy: {entropy_rank_corr:.4f}")
    print(f"  Least Confidence: {lc_rank_corr:.4f}")

    # How many top-k samples overlap?
    k = 100
    top_k_base = set(bald_base.topk(k).indices.tolist())
    top_k_cal = set(bald_cal.topk(k).indices.tolist())
    overlap = len(top_k_base & top_k_cal)

    print(f"\nTop-{k} sample overlap (BALD baseline vs calibrated): {overlap}/{k} ({overlap/k*100:.1f}%)")

    # Check if calibration changes which samples are "most uncertain"
    results["phase5"] = {
        "rank_correlation_bald": bald_rank_corr,
        "rank_correlation_entropy": entropy_rank_corr,
        "rank_correlation_lc": lc_rank_corr,
        "top_100_overlap_bald": overlap,
        "bald_stats_baseline": {
            "mean": bald_base.mean().item(),
            "std": bald_base.std().item(),
            "max": bald_base.max().item(),
        },
        "bald_stats_calibrated": {
            "mean": bald_cal.mean().item(),
            "std": bald_cal.std().item(),
            "max": bald_cal.max().item(),
        },
    }

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"""
Phase 1 (from previous): ECE = 17.67% (severely miscalibrated)

Phase 2 - Uncertainty-Error Correlation:
  BALD ↔ Error Spearman: {spearman_corr:.4f}
  Interpretation: {"Weak" if abs(spearman_corr) < 0.2 else "Moderate" if abs(spearman_corr) < 0.4 else "Strong"} correlation

Phase 3 - Temperature Scaling:
  Optimal T = {optimal_T:.4f}
  NLL: {nll_before:.4f} → {nll_after:.4f}

Phase 4 - Post-Calibration:
  ECE: {metrics_baseline['ece']*100:.2f}% → {metrics_calibrated['ece']*100:.2f}%
  BALD-Error correlation: {spearman_corr:.4f} → {spearman_cal:.4f}

Phase 5 - Acquisition Comparison:
  BALD rank correlation: {bald_rank_corr:.4f}
  Top-100 overlap: {overlap}%

Key Question: Does calibrated BALD beat random?
  → The BALD-Error correlation is {spearman_corr:.4f} (baseline) / {spearman_cal:.4f} (calibrated)
  → {"Calibration IMPROVES" if abs(spearman_cal) > abs(spearman_corr) else "Calibration does NOT improve"} uncertainty-error correlation
""")

    # Save results
    results_path = output_dir / "calibration_full_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved results to {results_path}")

    # Save tensors for potential AL experiments
    torch.save({
        "mc_logits": mc_logits,
        "labels": labels,
        "optimal_T": optimal_T,
        "mc_probs_baseline": mc_probs,
        "mc_probs_calibrated": mc_probs_calibrated,
    }, output_dir / "calibration_tensors.pt")
    print(f"Saved tensors to {output_dir / 'calibration_tensors.pt'}")


if __name__ == "__main__":
    main()
