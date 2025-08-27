import json
from pathlib import Path
from typing import List, Dict, Set
from torch.utils.data import DataLoader, ConcatDataset

from .datasets import PreferenceDataset, ALIndexedDataset
from .collate import collate_fn


class ActiveLearningDataManager:
    """Manages labeled/unlabeled state and creates DataLoaders for active learning."""

    def __init__(
        self,
        data_root: str,
        experiment_name: str = "al_experiment",
        num_workers: int = 4,
    ):
        self.data_root = Path(data_root)
        self.num_workers = num_workers

        self.state_dir = self.data_root / f"{experiment_name}_state"
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "al_state.json"

        self.seed_dataset = PreferenceDataset(self.data_root / "seed")
        self.pool_dataset = PreferenceDataset(self.data_root / "pool")
        self.test_dataset = PreferenceDataset(self.data_root / "test")

        self.labeled_pool_indices: Set[int] = set()
        self.current_iteration = 0

        if self.state_file.exists():
            self._load_state()
        else:
            self._init_state()

    def _init_state(self):
        self.labeled_pool_indices = set()
        self.current_iteration = 0
        self._save_state()
        print(
            f"Initialized AL Manager: Seed({len(self.seed_dataset)}), Pool({len(self.pool_dataset)}), Test({len(self.test_dataset)})"
        )

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(
                {
                    "labeled_pool_indices": list(self.labeled_pool_indices),
                    "current_iteration": self.current_iteration,
                },
                f,
            )

    def _load_state(self):
        with open(self.state_file, "r") as f:
            state = json.load(f)
        self.labeled_pool_indices = set(state["labeled_pool_indices"])
        self.current_iteration = state["current_iteration"]
        print(
            f"Loaded AL state: iteration {self.current_iteration}, pool labeled: {len(self.labeled_pool_indices)}"
        )

    def get_labeled_dataloader(
        self, batch_size: int = 16, shuffle: bool = True
    ) -> DataLoader:
        """DataLoader for all labeled data (seed + labeled pool)."""
        datasets = []

        seed_indices = list(range(len(self.seed_dataset)))
        datasets.append(ALIndexedDataset(self.seed_dataset, seed_indices))

        if self.labeled_pool_indices:
            pool_indices = list(self.labeled_pool_indices)
            datasets.append(ALIndexedDataset(self.pool_dataset, pool_indices))

        combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        return DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_unlabeled_dataloader(
        self, batch_size: int = 32, shuffle: bool = False
    ) -> DataLoader:
        """DataLoader for unlabeled pool data."""
        unlabeled_indices = [
            i
            for i in range(len(self.pool_dataset))
            if i not in self.labeled_pool_indices
        ]

        if not unlabeled_indices:
            empty_dataset = ALIndexedDataset(self.pool_dataset, [])
            return DataLoader(
                empty_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )

        unlabeled_dataset = ALIndexedDataset(self.pool_dataset, unlabeled_indices)

        return DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_validation_dataloader(self, batch_size: int = 32) -> DataLoader:
        """DataLoader for validation/test data."""
        test_indices = list(range(len(self.test_dataset)))
        test_dataset = ALIndexedDataset(self.test_dataset, test_indices)

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def label_samples(self, pool_indices: List[int]):
        """Mark pool samples as labeled."""
        if not pool_indices:
            return

        valid_indices = [
            idx
            for idx in pool_indices
            if 0 <= idx < len(self.pool_dataset)
            and idx not in self.labeled_pool_indices
        ]

        if not valid_indices:
            print("No valid indices to label")
            return

        self.labeled_pool_indices.update(valid_indices)
        print(
            f"Labeled {len(valid_indices)} samples. Total labeled: {len(self.labeled_pool_indices)}"
        )
        self._save_state()

    def next_iteration(self):
        self.current_iteration += 1
        self._save_state()

    def get_unlabeled_pool_indices(self) -> List[int]:
        return [
            i
            for i in range(len(self.pool_dataset))
            if i not in self.labeled_pool_indices
        ]

    def get_stats(self) -> Dict:
        total_labeled = len(self.seed_dataset) + len(self.labeled_pool_indices)
        total_available = len(self.seed_dataset) + len(self.pool_dataset)
        unlabeled_count = len(self.get_unlabeled_pool_indices())

        return {
            "iteration": self.current_iteration,
            "seed_labeled": len(self.seed_dataset),
            "pool_labeled": len(self.labeled_pool_indices),
            "pool_unlabeled": unlabeled_count,
            "total_labeled": total_labeled,
            "total_available": total_available,
            "progress": total_labeled / total_available if total_available > 0 else 0,
        }

    def has_unlabeled_data(self) -> bool:
        return len(self.get_unlabeled_pool_indices()) > 0
