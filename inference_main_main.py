import torch
from torchinfo import summary
from PIL import Image
from transformers import CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np
from trainer.models.clip_model import CLIPModel, ClipModelConfig

def main():
    device = "cpu"
    print(f"Using device: {device}")

    # 1. Initialize model configuration with MC dropout
    model_config = ClipModelConfig(
        pretrained_model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        dropout_rate=0.1,
        enable_mc_dropout=True,
    )

    # 2. Create the model
    model = CLIPModel(model_config)
    model = model.to(device)

    # 3. Load processor and configure image size
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    
    # 4. Set explicit image size - IMPORTANT: must be 224x224 based on the config
    IMAGE_SIZE = 224
    processor.image_processor.size = {"shortest_edge": IMAGE_SIZE, "longest_edge": IMAGE_SIZE}
    processor.image_processor.do_resize = True
    
    print(f"Processor configured with size: {processor.image_processor.size}")

    # 5. Load pretrained weights
    print("Loading pretrained weights...")
    state_dict = torch.load(
        "outputs/checkpoint-gstep200/pytorch_model.bin", weights_only=False
    )
    model.load_state_dict(state_dict)
    print("Weights loaded successfully")

    # 6. Summarize model architecture
    print("\nModel Architecture Summary:")
    summary(model, device=device)

    # 7. Check if dropout is present
    def count_dropout_layers(model):
        count = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                count += 1
        return count

    num_dropouts = count_dropout_layers(model)
    print(f"\nNumber of dropout layers found: {num_dropouts}")

    # 8. Prepare sample data for inference
    sample_images = [
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(73, 109, 137)),
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(220, 120, 80)),
    ]
    sample_prompt = "A beautiful landscape photo"

    # 9. MC Dropout inference
    print("\nRunning MC Dropout inference...")
    # Use the utility method for MC dropout inference
    mc_results = model.calc_probs_with_uncertainty(
        prompt=sample_prompt,
        images=sample_images,
        processor=processor,
        n_samples=30,
        device=device,
    )

    # Display results
    print(f"Mean probabilities: {mc_results['mean_probs']}")
    print(f"Standard deviations: {mc_results['std_probs']}")

    # 10. Visualize uncertainty (optional, if matplotlib is available)
    try:
        plt.figure(figsize=(10, 6))

        # Get all probability samples
        all_samples = np.array(mc_results["all_samples"])

        # Create violin plot to show distribution
        positions = np.arange(len(sample_images))
        plt.violinplot(
            [all_samples[:, i] for i in range(len(sample_images))], positions=positions
        )

        # Add mean and error bars
        plt.errorbar(
            positions,
            mc_results["mean_probs"],
            yerr=mc_results["std_probs"],
            fmt="o",
            color="black",
            capsize=5,
        )

        plt.xlabel("Image")
        plt.ylabel("Probability")
        plt.title("Preference Probability with Uncertainty (MC Dropout)")
        plt.xticks(positions, [f"Image {i + 1}" for i in range(len(sample_images))])

        plt.tight_layout()
        plt.savefig("mc_dropout_uncertainty.png")
        print("\nSaved uncertainty visualization to 'mc_dropout_uncertainty.png'")
    except Exception as e:
        print(f"Could not create visualization: {e}")

    # 11. Test if MC dropout is actually applying stochasticity
    print("\nTesting if MC dropout is applying stochasticity...")
    model.enable_mc_dropout = True
    model.eval()
    
    # Process text inputs for testing
    text_inputs = processor(
        text=sample_prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # Get features twice
        text_features1 = model.get_text_features(**text_inputs)
        text_features2 = model.get_text_features(**text_inputs)

        # Check if they're different (which they should be with MC dropout)
        is_different = not torch.allclose(text_features1, text_features2)

    print(f"MC dropout is working: {is_different}")

if __name__ == "__main__":
    main()