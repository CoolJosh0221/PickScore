import io
import json
import random
from pathlib import Path
from typing import Dict

from datasets import load_dataset, Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# ---------------- configs ----------------

CONFIGS: Dict[str, Dict] = {
    "quick": {  # small dataset for testing
        "target_samples": 1000,
        "splits": {"train": 0.8, "valid": 0.1, "test": 0.1},
        "seed": 123,
    },
    "prod": {  # full dataset for training
        "target_samples": 40000,
        "splits": {"train": 0.98, "valid": 0.01, "test": 0.01},
        "seed": 42,
    },
}


def sample_and_save(
    config: str,
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    split: str = "train",
    out_root: str = "./runs",
) -> Path:
    """
    Sample from HF dataset, create train/valid/test splits, and save each split
    as a HuggingFace dataset on disk under runs/<config>/<split>/.
    """
    cfg = CONFIGS[config]
    random.seed(cfg["seed"])

    out_dir = Path(out_root) / config
    out_dir.mkdir(parents=True, exist_ok=True)

    # streaming + shuffle = bounded memory
    ds: IterableDataset = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.filter(lambda x: x["has_label"])
    ds = ds.shuffle(buffer_size=1000, seed=cfg["seed"])

    total = cfg["target_samples"]
    # target counts per split
    targets = {k: int(v * total) for k, v in cfg["splits"].items()}
    # fix rounding drift
    diff = total - sum(targets.values())
    if diff:
        targets["train"] += diff

    taken = {k: 0 for k in targets}
    buffers = {k: [] for k in targets}

    for ex in ds:
        if sum(taken.values()) >= total:
            break
        s = random.choices(list(targets.keys()), weights=list(cfg["splits"].values()))[
            0
        ]
        if taken[s] >= targets[s]:
            continue
        buffers[s].append(ex)
        taken[s] += 1

    # save splits
    for s in targets:
        split_dir = out_dir / s
        split_dir.mkdir(parents=True, exist_ok=True)
        if buffers[s]:
            HFDataset.from_list(buffers[s]).save_to_disk(str(split_dir))
        # write a tiny manifest
        (split_dir / "manifest.json").write_text(
            json.dumps({"config": config, "split": s, "samples": taken[s]}, indent=2),
            "utf-8",
        )

    return out_dir


class PreferenceDataset(Dataset):
    """
    Wrap a HuggingFace dataset split (train/valid/test) for PyTorch training.
    Decodes images and returns tensors with labels.
    """

    def __init__(self, split_dir: Path, image_size: int = 512):
        self.ds = HFDataset.load_from_disk(str(split_dir))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        caption = row["caption"]
        img0 = Image.open(io.BytesIO(row["jpg_0"]))
        img1 = Image.open(io.BytesIO(row["jpg_1"]))
        lab0 = float(row["label_0"])
        lab1 = float(row["label_1"])
        return {
            "caption": caption,
            "image_0": img0,
            "image_1": img1,
            "label_0": torch.tensor(lab0, dtype=torch.float32),
            "label_1": torch.tensor(lab1, dtype=torch.float32),
        }


def create_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    image_size: int = 512,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a given split (train/valid/test).
    """
    ds = PreferenceDataset(Path(data_dir) / split, image_size=image_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
