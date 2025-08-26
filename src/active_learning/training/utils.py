import json
from datetime import datetime
from pathlib import Path
import random
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(base: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_root = (base / "checkpoints" / run_id).resolve()
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
