from pathlib import Path
import argparse
import io
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from active_learning.acquisition import (
    BALD,
    CoreSet,
    MaximalEntropy,
    MeanStdDev,
    Random,
)
from active_learning.engine import ActiveLearningEngine
from active_learning.utils.collate import collate_fn
from active_learning.models.clip_model_mcdo import MCDropoutCLIPModel
from active_learning.utils.data_utils import get_data_splits
from active_learning.utils.plotting import plot_learning_curves


def get_mcdo_prediction_fn(model, processor, device, num_samples):
    """Create MC dropout prediction function."""
    from active_learning.engine import get_preference_scores

    @torch.no_grad()
    def predict(batch):
        mc_probs = []
        with model.mc_eval():
            for _ in range(num_samples):
                score_0, score_1 = get_preference_scores(
                    model, processor, batch, device
                )
                prob = torch.sigmoid(score_0 - score_1)
                mc_probs.append(prob.unsqueeze(-1))
        return torch.cat(mc_probs, dim=-1)

    return predict


def get_feature_extractor_fn(model, processor, device):
    """Create feature extraction function."""
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    @torch.no_grad()
    def extract_features(dataloader):
        all_features = []
        model.eval()

        for batch in tqdm(dataloader, desc="Feature Extraction"):
            images_0 = [Image.open(io.BytesIO(image)) for image in batch["jpg_0"]]
            images_1 = [Image.open(io.BytesIO(image)) for image in batch["jpg_1"]]
            images = images_0 + images_1

            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = F.normalize(
                model.get_image_features(**image_inputs), p=2, dim=-1
            )

            batch_size = len(batch["jpg_0"])
            avg_features = (
                image_features[:batch_size] + image_features[batch_size:]
            ) / 2.0
            all_features.append(avg_features)

        return torch.cat(all_features, dim=0)

    return extract_features


def setup_gpu():
    """Setup GPU environment."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def main():
    parser = argparse.ArgumentParser(
        description="Active Learning for CLIP Reward Modeling"
    )

    # Data
    parser.add_argument("--seed_size", type=int, default=200)
    parser.add_argument("--pool_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--random_seed", type=int, default=42)

    # Model
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--mc_dropout_p", type=float, default=0.1)

    # Training
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    # Acquisition
    parser.add_argument(
        "--acquisition_strategy",
        choices=["bald", "coreset", "random", "max_entropy", "mean_std_dev"],
        default="bald",
    )
    parser.add_argument("--al_iterations", type=int, default=10)
    parser.add_argument("--acquisition_batch_size", type=int, default=100)
    parser.add_argument("--num_mc_samples", type=int, default=20)

    # Experiment
    parser.add_argument("--wandb_project", default="active-learning")
    parser.add_argument("--eval_with_mc_dropout", action="store_true")

    # Plotting
    parser.add_argument("--plot_run_ids", nargs="+")

    args = parser.parse_args()

    # Handle plotting mode
    if args.plot_run_ids:
        plot_learning_curves(
            args.plot_run_ids, args.wandb_project, "learning_curves.png"
        )
        return

    # Setup
    device = setup_gpu()
    print(f"Using device {device}")
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"{args.acquisition_strategy}_seed{args.seed_size}",
    )

    # Data
    cache_dir= Path.cwd() / "dataset_cache"
    print(f"Cache dir: {cache_dir}")
    splits = get_data_splits(
        seed_size=args.seed_size,
        pool_size=args.pool_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        cache_dir=cache_dir
    )

    labeled_set, unlabeled_pool = splits["seed"], splits["pool"]
    test_loader = DataLoader(
        splits["test"],
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = MCDropoutCLIPModel(
        pretrained_model_name_or_path=args.model_name, mc_dropout_p=args.mc_dropout_p
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Strategy
    if args.acquisition_strategy in ["bald", "max_entropy", "mean_std_dev"]:
        pred_fn = get_mcdo_prediction_fn(
            model, model.processor, device, args.num_mc_samples
        )
        strategy_map = {
            "bald": BALD,
            "max_entropy": MaximalEntropy,
            "mean_std_dev": MeanStdDev,
        }
        strategy = strategy_map[args.acquisition_strategy](prediction_fn=pred_fn)
    elif args.acquisition_strategy == "coreset":
        feature_fn = get_feature_extractor_fn(model, model.processor, device)
        strategy = CoreSet(feature_extractor_fn=feature_fn)
    else:  # random
        strategy = Random()

    # Run
    engine = ActiveLearningEngine(
        model, optimizer, model.processor, strategy, device, args
    )
    engine.run(labeled_set, unlabeled_pool, test_loader)

    wandb.finish()


if __name__ == "__main__":
    main()
