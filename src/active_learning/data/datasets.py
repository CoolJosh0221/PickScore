import io
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class PreferenceDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.ds = HFDataset.load_from_disk(str(split_dir))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return {
            "caption": row["caption"],
            "image_0": Image.open(io.BytesIO(row["jpg_0"])).convert("RGB"),
            "image_1": Image.open(io.BytesIO(row["jpg_1"])).convert("RGB"),
            "label_0": torch.tensor(float(row["label_0"]), dtype=torch.float32),
            "label_1": torch.tensor(float(row["label_1"]), dtype=torch.float32),
        }
