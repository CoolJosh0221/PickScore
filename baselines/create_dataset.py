from active_learning.data.sampling import create_standard_splits

seed = 45510

ds_dir = "baselines/dataset/"

# TODO: Increase data size after functionality testing
ds_dir_path = create_standard_splits(
    ds_dir,
    dataset_name="pickapic-anonymous/pickapic_v1",
    train_size=8000,
    val_size=1000,
    test_size=1000,
    seed=seed,
)

print(f"Dataset created at {ds_dir_path}")
