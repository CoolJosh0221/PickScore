from active_learning.data.sampling import sample_and_save

out = sample_and_save(
    "./runs/quick",
    dataset_name="pickapic-anonymous/pickapic_v1",
    total_samples=5000,
    split_fractions={
        "seed": 0.05,
        "pool": 0.75,
        "valid": 0.10,
        "test": 0.10,
    },
    split="train"
)
