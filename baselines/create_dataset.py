import os

from datasets import load_from_disk
from dotenv import load_dotenv
import huggingface_hub

from active_learning.data.sampling import create_standard_splits

load_dotenv()

huggingface_hub.login(token=os.getenv("HF_TOKEN"))

seed = 45510

ds_dir = "baselines/dataset/"

ds_dir_path = create_standard_splits(
    ds_dir,
    dataset_name="pickapic-anonymous/pickapic_v1",
    train_size=8000,
    val_size=1000,
    test_size=1000,
    seed=seed,
)

print(f"Dataset created at {ds_dir_path}")

ds = load_from_disk(ds_dir_path)
ds.push_to_hub("cooljosh0221/pickapic-10000")
