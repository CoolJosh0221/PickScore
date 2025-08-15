from pathlib import Path
from itertools import islice
from typing import Any, Iterable, List, Mapping
import hashlib
import json

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from tqdm.auto import tqdm


def _stream_and_filter(
    split_name: str, dataset_name: str, random_seed: int
) -> IterableDataset:
    """Return a shuffled, filtered streaming split where only examples with 'are_different' == True are kept."""
    stream: IterableDataset = load_dataset(
        dataset_name, split=split_name, streaming=True
    )
    stream = stream.shuffle(
        seed=random_seed, buffer_size=10000
    )  # Larger buffer for better shuffling
    return stream.filter(lambda x: bool(x.get("are_different", False)))


def _take_n(
    stream: Iterable[Mapping[str, Any]], n: int, desc: str
) -> List[Mapping[str, Any]]:
    """Take exactly `n` examples from a streaming dataset, raising if the stream ends prematurely."""
    items = list(tqdm(islice(stream, n), total=n, desc=desc))
    if len(items) < n:
        raise ValueError(
            f"Stream ended before collecting {n} samples; collected {len(items)}."
        )
    return items


def _get_cache_key(
    dataset_name: str, seed_size: int, pool_size: int, test_size: int, random_seed: int
) -> str:
    """Generate a unique cache key based on parameters."""
    param_str = f"{dataset_name}_{seed_size}_{pool_size}_{test_size}_{random_seed}"
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def get_data_splits(
    dataset_name: str = "pickapic-anonymous/pickapic_v1",
    seed_size: int = 200,
    pool_size: int = 10_000,
    test_size: int = 2_000,
    random_seed: int = 42,
    cache_dir: str | Path = "./dataset_cache",
    use_cache: bool = True,
    force_refresh: bool = False,
) -> DatasetDict:
    """
    Build or load cached DatasetDict splits ('seed', 'pool', 'test') from a streaming dataset.

    This function streams from Hugging Face datasets to avoid downloading the full corpus,
    shuffles deterministically, filters out samples where 'are_different' is False, and then
    materializes a fixed number of examples for each split. The resulting DatasetDict can be
    cached to disk for reuse across runs.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        seed_size: Number of examples in the seed (initial training) set from the 'train' split.
        pool_size: Number of examples in the unlabeled pool from the 'train' split.
        test_size: Number of examples in the fixed test set from the 'validation' split.
        random_seed: Base seed for deterministic shuffling; validation uses seed+1.
        cache_dir: Directory for saving/loading the cached DatasetDict.
        use_cache: Whether to attempt loading from and saving to disk cache.
        force_refresh: If True, rebuild splits even if cache is present.

    Returns:
        DatasetDict containing three Dataset objects: 'seed', 'pool', and 'test'.
    """
    cache_key = _get_cache_key(
        dataset_name, seed_size, pool_size, test_size, random_seed
    )
    cache_path = Path(cache_dir) / f"splits_{cache_key}"

    # Check for existing cache
    if use_cache and not force_refresh and cache_path.exists():
        try:
            dd = DatasetDict.load_from_disk(str(cache_path))

            # Verify cache integrity
            cache_info_path = cache_path / "cache_info.json"
            if cache_info_path.exists():
                with open(cache_info_path, "r") as f:
                    cache_info = json.load(f)

                expected_info = {
                    "dataset_name": dataset_name,
                    "seed_size": seed_size,
                    "pool_size": pool_size,
                    "test_size": test_size,
                    "random_seed": random_seed,
                }

                if cache_info == expected_info and all(
                    split in dd and len(dd[split]) == size
                    for split, size in [
                        ("seed", seed_size),
                        ("pool", pool_size),
                        ("test", test_size),
                    ]
                ):
                    print(f"✓ Loaded cached DatasetDict from {cache_path}")
                    return dd

            print("Cache present but integrity check failed; rebuilding")

        except Exception as e:
            print(f"Failed to load cache ({e}); rebuilding")

    print(f"Building splits from streaming datasets (cache key: {cache_key})")

    # Build train splits
    print("Fetching training data...")
    train_stream = _stream_and_filter("train", dataset_name, random_seed)
    train_needed = seed_size + pool_size
    train_samples = _take_n(train_stream, train_needed, desc="Fetching train samples")

    seed_ds = Dataset.from_list(train_samples[:seed_size])
    pool_ds = Dataset.from_list(train_samples[seed_size:])

    # Build validation split
    print("Fetching validation data...")
    val_stream = _stream_and_filter("validation", dataset_name, random_seed + 1)
    test_samples = _take_n(val_stream, test_size, desc="Fetching validation samples")
    test_ds = Dataset.from_list(test_samples)

    dd = DatasetDict({"seed": seed_ds, "pool": pool_ds, "test": test_ds})

    # Save to cache if requested
    if use_cache:
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            dd.save_to_disk(str(cache_path))

            # Save cache metadata
            cache_info = {
                "dataset_name": dataset_name,
                "seed_size": seed_size,
                "pool_size": pool_size,
                "test_size": test_size,
                "random_seed": random_seed,
            }

            with open(cache_path / "cache_info.json", "w") as f:
                json.dump(cache_info, f, indent=2)

            print(f"✓ Saved DatasetDict to {cache_path}")
        except Exception as e:
            print(f"Warning: failed to save cache ({e})")

    print(
        f"Data splits ready: seed={len(dd['seed'])}, pool={len(dd['pool'])}, test={len(dd['test'])}"
    )

    # Print sample statistics
    print(f"Sample captions length stats:")
    sample_texts = [item["caption"] for item in dd["seed"][:100]]
    lengths = [len(text.split()) for text in sample_texts]
    print(f"  Mean: {sum(lengths) / len(lengths):.1f} words")
    print(f"  Min/Max: {min(lengths)}/{max(lengths)} words")

    return dd


def validate_dataset_split(dataset_dict: DatasetDict) -> bool:
    """
    Validate that a dataset split has the expected structure and content.

    Returns:
        bool: True if validation passes, False otherwise.
    """
    required_keys = {"seed", "pool", "test"}
    if not required_keys.issubset(set(dataset_dict.keys())):
        print(
            f"Missing required splits. Expected: {required_keys}, Got: {set(dataset_dict.keys())}"
        )
        return False

    required_columns = {
        "caption",
        "jpg_0",
        "jpg_1",
        "label_0",
        "label_1",
        "are_different",
    }
    for split_name, split_data in dataset_dict.items():
        if not required_columns.issubset(set(split_data.column_names)):
            print(f"Split '{split_name}' missing required columns.")
            print(f"Expected: {required_columns}")
            print(f"Got: {set(split_data.column_names)}")
            return False

        # Check that all samples have are_different=True
        if not all(item.get("are_different", False) for item in split_data):
            print(f"Split '{split_name}' contains samples with are_different=False")
            return False

    print("✓ Dataset validation passed")
    return True
