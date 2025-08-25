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
    total_samples: int,
    split_fractions: Mapping[str, float],
    split: str = "train",
    seed: int = 42,
    shuffle_buffer: int = 1000,
) -> Path:
    random.seed(seed)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    stream: IterableDataset = load_dataset(dataset_name, split=split, streaming=True)
    stream = stream.filter(lambda x: x["are_different"])
    stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)

    keys = list(split_fractions.keys())
    taken = {k: 0 for k in keys}
    buffers = {k: [] for k in keys}
    weights = [split_fractions[k] if split_fractions is not None else 1.0 for k in keys]

    for ex in stream:
        if sum(taken.values()) >= total_samples:
            break
        s = random.choices(keys, weights=weights, k=1)[0]
        buffers[s].append(ex)
        taken[s] += 1

    for k in keys:
        split_dir = out_dir / k
        split_dir.mkdir(parents=True, exist_ok=True)
        if buffers[k]:
            HFDataset.from_list(buffers[k]).save_to_disk(str(split_dir))
        (split_dir / "manifest.json").write_text(
            json.dumps({"split": k, "samples": taken[k]}, indent=2),
            "utf-8",
        )

    return out_dir
