import json
from pathlib import Path
from typing import Mapping, Optional
from datasets import load_dataset, Dataset as HFDataset
from datasets.iterable_dataset import IterableDataset
from datasets.arrow_writer import ArrowWriter
import pyarrow as pa  # Verify dep


def sample_and_save(
    out_root: str,
    *,
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    split_sizes: Mapping[str, int],
    split: str = "train",
    seed: int = 42,
    shuffle_buffer: int = 1000,
    hf_hub_name: Optional[str] = None,
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

    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if any(out_dir.iterdir()):
        print(f"Dataset already exists at {out_dir}, skipping...")

    # Load and prepare streaming dataset
    stream: IterableDataset = load_dataset(dataset_name, split=split, streaming=True)  # type: ignore
    stream = stream.filter(lambda x: x["are_different"])
    stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)

    for key, size in split_sizes.items():
        split_dir = out_dir / key
        split_dir.mkdir(parents=True, exist_ok=True)
        sampled = stream.take(size)

        ds = HFDataset.from_generator(
            lambda: (yield from sampled),
            features=sampled.features,
        )
        ds.save_to_disk(split_dir)

    # TODO: Impl pushing to hf hub

    """
	features = stream.features  # required for ArrowWriter

	# One ArrowWriter per split
	writers = {}
	for k in keys:
		split_dir = out_dir / k
		split_dir.mkdir(parents=True, exist_ok=True)
		writers[k] = ArrowWriter(
			path=str(split_dir / f"{k}.arrow"),
			features=features,
			writer_batch_size=chunk_size,
		)

	# Fill splits sequentially since the stream is already shuffled
	it = iter(stream)
	try:
		for k in keys:
			target = split_sizes[k]
			for _ in range(target):
				ex = next(it)
				writers[k].write(ex)
				taken[k] += 1
	except StopIteration:
		pass  # source exhausted

	# Save dataset to disk
	for k in keys:
		split_dir = out_dir / k
		writers[k].finalize()
		ds = HFDataset.from_file(str(split_dir / f"{k}.arrow"))
		ds.save_to_disk(split_dir)
		try:
			(split_dir / f"{k}.arrow").unlink()
		except Exception:
			pass

		# Per-split manifest
		(split_dir / "manifest.json").write_text(
			json.dumps(
				{
					"split": k,
					"samples": taken[k],
					"target": split_sizes[k],
					"completed": taken[k] == split_sizes[k],
					"format": "hf_save_to_disk",
				},
				indent=2,
			),
			"utf-8",
		)

	# Save summary manifest
	(out_dir / "summary.json").write_text(
		json.dumps(
			{
				"total_requested": sum(split_sizes.values()),
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
	"""

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


def create_independent_test_set(
    out_root: str,
    *,
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    size: int = 3000,
    seed: int = 42,
):
    return sample_and_save(
        out_root,
        dataset_name=dataset_name,
        split_sizes={"test": size},
        seed=seed,
    )
