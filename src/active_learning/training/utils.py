import json
import math
from datetime import datetime
from pathlib import Path
import random
from typing import Optional
import torch
from torch.optim.lr_scheduler import LambdaLR


class EarlyStopping:
    """Early stopping to prevent overfitting during training.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in monitored value to qualify as improvement.
        mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better).
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        """Check if training should stop.

        Args:
            current_score: Current validation metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improved = current_score < self.best_score - self.min_delta
        else:  # mode == "max"
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state for a new training run."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(base: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_root = (base / run_id).resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)
    return ckpt_root


def save_epoch(
    model, out_dir: Path, ckpt_root: Path, epoch: int, metrics: dict
) -> Path:
    tag = f"epoch-{epoch:04d}_overall-{metrics['overall_acc']:.4f}"
    ep_dir = ckpt_root / tag
    ep_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ep_dir / "model.pth"))  # Save as file inside directory
    (ep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "LAST").write_text(str(ep_dir))
    return ep_dir


def save_best_pointer(out_dir: Path, best_path: Path) -> None:
    (out_dir / "BEST").write_text(str(best_path))


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate as a ratio of initial lr (default 0.1).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay to min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)


def reset_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create a fresh AdamW optimizer with reset state.

    Args:
        model: The model whose parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay.

    Returns:
        A new AdamW optimizer instance with fresh state.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
