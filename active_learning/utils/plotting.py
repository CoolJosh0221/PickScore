import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

from wandb.errors import CommError


def plot_learning_curves(
    run_ids: List[str], 
    project_name: str, 
    output_path: str,
    title: Optional[str] = None,
    metrics: List[str] = ["eval_accuracy"],
    figsize: tuple = (12, 8)
) -> str:
    """
    Fetches run data from W&B and plots learning curves to compare strategies.

    Args:
        run_ids: A list of W&B run IDs to compare.
        project_name: The W&B project name.
        output_path: Path to save the plot image.
        title: Custom title for the plot.
        metrics: List of metrics to plot.
        figsize: Figure size tuple.
        
    Returns:
        Path to the saved plot.
    """
    api = wandb.Api()
    all_runs_df = []

    print(f"Fetching data for {len(run_ids)} runs...")
    
    for i, run_id in enumerate(run_ids):
        try:
            run = api.run(f"{project_name}/{run_id}")
            
            # Fetch the relevant keys for plotting
            keys_to_fetch = ["labeled_set_size", "al_iteration"] + metrics
            history = run.history(keys=keys_to_fetch)
            
            if history.empty:
                print(f"Warning: No data found for run {run_id}")
                continue
                
            # Add config values to the dataframe for grouping and labeling
            config = run.config
            history["strategy"] = config.get("acquisition_strategy", "unknown")
            history["model_type"] = config.get("model_type", "mcdo")
            history["seed_size"] = config.get("seed_size", "unknown")
            history["mc_dropout_p"] = config.get("mc_dropout_p", "N/A")
            history["run_id"] = run_id[:8]  # Short run ID for legend
            
            # Create a combined label for the legend
            if history["mc_dropout_p"].iloc[0] != "N/A":
                history["legend_label"] = (
                    f"{history['strategy']} (p={history['mc_dropout_p'].iloc[0]:.2f})"
                )
            else:
                history["legend_label"] = history["strategy"]
                
            all_runs_df.append(history)
            print(f"✓ Fetched data for run {i+1}/{len(run_ids)}: {run_id[:8]}")
            
        except CommError as e:
            print(f"Could not fetch run {run_id}. Error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error fetching run {run_id}: {e}")
            continue

    if not all_runs_df:
        print("No run data found. Cannot generate plot.")
        return output_path

    full_df = pd.concat(all_runs_df, ignore_index=True)
    
    # Set up the plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # Create subplots for multiple metrics
    n_metrics = len(metrics)
    if n_metrics == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_metrics, 1, figsize=(figsize[0], figsize[1] * n_metrics))
        if n_metrics == 1:
            axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Filter out NaN values for this metric
        metric_df = full_df.dropna(subset=[metric])
        
        if metric_df.empty:
            print(f"Warning: No data found for metric {metric}")
            continue
            
        # Plot using seaborn for clean look
        sns.lineplot(
            data=metric_df,
            x="labeled_set_size",
            y=metric,
            hue="legend_label",
            style="seed_size",
            marker="o",
            ax=ax,
            markersize=6,
            linewidth=2
        )
        
        # Customize the subplot
        ax.set_xlabel("Number of Labeled Samples", fontsize=11)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        
        # Move legend outside the plot area
        if i == 0:  # Only show legend on first subplot
            ax.legend(title="Strategy / Seed Size", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.get_legend().remove()

    # Set overall title
    if title is None:
        title = "Active Learning Performance Comparison"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Learning curve plot saved to {output_path}")
    return output_path


def plot_uncertainty_analysis(
    run_ids: List[str],
    project_name: str,
    output_path: str,
    uncertainty_metrics: List[str] = ["train_loss"]
) -> str:
    """
    Plot uncertainty-related metrics to analyze model confidence over time.
    
    Args:
        run_ids: List of W&B run IDs to compare.
        project_name: The W&B project name.
        output_path: Path to save the plot.
        uncertainty_metrics: Metrics related to uncertainty to plot.
        
    Returns:
        Path to the saved plot.
    """
    api = wandb.Api()
    all_runs_df = []

    for run_id in run_ids:
        try:
            run = api.run(f"{project_name}/{run_id}")
            keys_to_fetch = ["labeled_set_size", "al_iteration"] + uncertainty_metrics
            history = run.history(keys=keys_to_fetch)
            
            if not history.empty:
                history["strategy"] = run.config.get("acquisition_strategy", "unknown")
                history["mc_dropout_p"] = run.config.get("mc_dropout_p", 0.1)
                all_runs_df.append(history)
                
        except Exception as e:
            print(f"Error fetching uncertainty data for run {run_id}: {e}")
            continue

    if not all_runs_df:
        print("No uncertainty data found.")
        return output_path

    full_df = pd.concat(all_runs_df, ignore_index=True)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(uncertainty_metrics), 1, figsize=(12, 6 * len(uncertainty_metrics)))
    
    if len(uncertainty_metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(uncertainty_metrics):
        ax = axes[i]
        metric_df = full_df.dropna(subset=[metric])
        
        sns.lineplot(
            data=metric_df,
            x="labeled_set_size",
            y=metric,
            hue="strategy",
            style="mc_dropout_p",
            marker="o",
            ax=ax
        )
        
        ax.set_xlabel("Number of Labeled Samples")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Dataset Size")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"✓ Uncertainty analysis plot saved to {output_path}")
    return output_path


def create_summary_table(run_ids: List[str], project_name: str, output_path: str) -> str:
    """
    Create a summary table of final performance metrics for each run.
    
    Args:
        run_ids: List of W&B run IDs to summarize.
        project_name: The W&B project name.
        output_path: Path to save the table (CSV format).
        
    Returns:
        Path to the saved table.
    """
    api = wandb.Api()
    summary_data = []

    for run_id in run_ids:
        try:
            run = api.run(f"{project_name}/{run_id}")
            
            # Get final metrics
            history = run.history(keys=["eval_accuracy", "train_loss", "labeled_set_size"])
            final_metrics = history.iloc[-1] if not history.empty else {}
            
            config = run.config
            summary_row = {
                "run_id": run_id,
                "strategy": config.get("acquisition_strategy", "unknown"),
                "seed_size": config.get("seed_size", "unknown"),
                "mc_dropout_p": config.get("mc_dropout_p", "N/A"),
                "final_accuracy": final_metrics.get("eval_accuracy", "N/A"),
                "final_train_loss": final_metrics.get("train_loss", "N/A"),
                "final_labeled_size": final_metrics.get("labeled_set_size", "N/A"),
                "run_name": run.name,
            }
            summary_data.append(summary_row)
            
        except Exception as e:
            print(f"Error fetching summary for run {run_id}: {e}")
            continue

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"✓ Summary table saved to {output_path}")
        
        # Print top performers
        if "final_accuracy" in df.columns and df["final_accuracy"].notna().any():
            numeric_acc = pd.to_numeric(df["final_accuracy"], errors="coerce")
            top_performer = df.loc[numeric_acc.idxmax()]
            print(f"Top performer: {top_performer['strategy']} "
                  f"(accuracy: {top_performer['final_accuracy']:.4f})")
    
    return output_path