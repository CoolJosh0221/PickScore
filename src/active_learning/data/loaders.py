from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .datasets import PreferenceDataset
from .collate import collate_fn


def create_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    ds = PreferenceDataset(Path(data_dir) / split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )
