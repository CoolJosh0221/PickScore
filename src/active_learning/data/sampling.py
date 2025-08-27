import json
import random
from pathlib import Path
from typing import Mapping
from datasets import load_dataset, Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset


def sample_and_save(
    out_root: str,
    *,
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    split_sizes: Mapping[str, int],  # Changed from split_fractions to split_sizes
    split: str = "train",
    seed: int = 42,
    shuffle_buffer: int = 1000,
) -> Path:
    """
    Sample and save dataset splits by exact sample counts.

    Args:
        out_root: Output directory path
        dataset_name: Name of the dataset to load
        split_sizes: Mapping of split names to exact number of samples wanted
                    e.g., {"train": 1000, "val": 200, "test": 300}
        split: Source split to sample from
        seed: Random seed for reproducibility
        shuffle_buffer: Buffer size for shuffling

    Returns:
        Path to output directory
    """
    random.seed(seed)

    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare streaming dataset
    stream: IterableDataset = load_dataset(dataset_name, split=split, streaming=True)  # type: ignore
    stream = stream.filter(lambda x: x["are_different"])
    stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)

    keys = list(split_sizes.keys())
    taken = {k: 0 for k in keys}
    buffers = {k: [] for k in keys}

    # Calculate total samples needed
    total_samples = sum(split_sizes.values())

    for ex in stream:
        # Check if we've collected all samples
        if sum(taken.values()) >= total_samples:
            break

        # Find splits that haven't reached their target size
        available_splits = [k for k in keys if taken[k] < split_sizes[k]]

        if not available_splits:
            break  # All splits are full

        # Weight by remaining samples needed for balanced distribution
        remaining = {k: split_sizes[k] - taken[k] for k in available_splits}
        weights = list(remaining.values())

        # Choose split based on remaining samples needed
        s = random.choices(available_splits, weights=weights, k=1)[0]
        buffers[s].append(ex)
        taken[s] += 1

    # Save splits to disk
    for k in keys:
        split_dir = out_dir / k
        split_dir.mkdir(parents=True, exist_ok=True)

        if buffers[k]:
            HFDataset.from_list(buffers[k]).save_to_disk(str(split_dir))

        # Save manifest with actual and target counts
        (split_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "split": k,
                    "samples": taken[k],
                    "target": split_sizes[k],
                    "completed": taken[k] == split_sizes[k],
                },
                indent=2,
            ),
            "utf-8",
        )

    # Save summary manifest
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_requested": total_samples,
                "total_collected": sum(taken.values()),
                "splits": {
                    k: {"collected": taken[k], "target": split_sizes[k]} for k in keys
                },
                "seed": seed,
                "dataset_name": dataset_name,
                "source_split": split,
            },
            indent=2,
        ),
        "utf-8",
    )

    return out_dir


def create_active_learning_splits(
    out_root: str,
    *,
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    seed_size: int = 500,
    pool_size: int = 10000,
    test_size: int = 2000,
    seed: int = 42,
) -> Path:
    """Create splits for active learning experiments."""
    return sample_and_save(
        out_root,
        dataset_name=dataset_name,
        split_sizes={
            "seed": seed_size,  # Initial labeled set
            "pool": pool_size,  # Unlabeled pool for active learning
            "test": test_size,  # Test set for evaluation
        },
        seed=seed,
    )


def create_standard_splits(
    out_root: str,
    *,
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    seed: int = 42,
) -> Path:
    """Create standard train/val/test splits."""
    return sample_and_save(
        out_root,
        dataset_name=dataset_name,
        split_sizes={"train": train_size, "val": val_size, "test": test_size},
        seed=seed,
    )
