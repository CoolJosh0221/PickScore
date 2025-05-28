import torch
from torchinfo import summary
from PIL import Image
from transformers import CLIPProcessor
import matplotlib.pyplot as plt
import numpy as np
from trainer.models.clip_model import CLIPModel, ClipModelConfig

# # --- put this at the very top of your inference.py, before any model imports/instantiation ---
# from transformers.models.clip.modeling_clip import CLIPTextEmbeddings

# # 1) keep a reference to the original
# _orig_clip_text_embeddings_forward = CLIPTextEmbeddings.forward


# # 2) define a new forward with the **same signature**
# def debug_forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
#     # if no position_ids were passed in, recreate them exactly as HF would
#     if position_ids is None:
#         bsz, seq_len = input_ids.shape
#         position_ids = (
#             torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
#             .unsqueeze(0)
#             .expand(bsz, -1)
#         )

#     # print what you're actually getting
#     print(
#         f"→ position_ids range: {position_ids.min().item()} … {position_ids.max().item()}"
#     )
#     print(f"→ max_position_embeddings: {self.position_embedding.num_embeddings}")

#     # 3) call the original implementation
#     return _orig_clip_text_embeddings_forward(
#         self,
#         input_ids=input_ids,
#         position_ids=position_ids,
#         inputs_embeds=inputs_embeds,
#     )


# # 4) patch it in
# CLIPTextEmbeddings.forward = debug_forward
# # ------------------------------------------------------------------------------

# from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

# # Keep original method reference
# _orig_vision_embeddings_forward = CLIPVisionEmbeddings.forward

# # Define debug forward method for vision embeddings
# def vision_debug_forward(self, pixel_values):
#     batch_size = pixel_values.shape[0]
#     patch_embeds = self.patch_embedding(pixel_values)

#     # Print helpful debug info
#     num_patches = patch_embeds.shape[1]
#     print(f"→ Vision: num patches: {num_patches}")
#     print(f"→ Vision: position embedding size: {self.position_embedding.num_embeddings}")

#     # Check if we're using too many patches for the position embedding table
#     if num_patches + 1 > self.position_embedding.num_embeddings:  # +1 for class token
#         raise ValueError(
#             f"Too many patches ({num_patches}+1) for position embeddings table size "
#             f"({self.position_embedding.num_embeddings}). Try reducing image size."
#         )

#     # Call original with extra safety
#     return _orig_vision_embeddings_forward(self, pixel_values)

# # Patch in the debug method
# CLIPVisionEmbeddings.forward = vision_debug_forward


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")

    # 1. Initialize model configuration with MC dropout
    model_config = ClipModelConfig(
        pretrained_model_name_or_path="yuvalkirstain/PickScore_v1",
        # dropout_rate=0.1,
        # enable_mc_dropout=False,  # Start with MC dropout disabled
    )

    # 2. Create the model
    model = CLIPModel(model_config)
    model = model.to(device)

    # 3. Load pretrained weights
    # print("Loading pretrained weights...")
    # state_dict = torch.load(
    #     "outputs/checkpoint-gstep100/pytorch_model.bin", weights_only=False
    # )
    # model.load_state_dict(state_dict)
    # print("Weights loaded successfully")

    print(
        "pos-embedding weight shape:",
        model.model.text_model.embeddings.position_embedding.weight.shape,
    )
    print(
        "max_position_embeddings in config:",
        model.model.config.text_config.max_position_embeddings,
    )

    # 4. Summarize model architecture
    # print("\nModel Architecture Summary:")

    # # Get summary
    # summary(model, device=device)

    # 5. Load processor for inference
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    print(f"Processor name: {processor.__class__.__name__}")
    print(f"Processor image size: {processor.image_processor.size}")
    print(f"Do not resize in processor: {processor.image_processor.do_resize}")

    # 6. Prepare sample data for inference
    # For demo purposes, create two random images - replace with actual images
    sample_images = [
        Image.open("skyline_image.jpg"),
    ]
    sample_prompt = "A beautiful photo of a night city skyline in Tokyo"

    # 7. Standard inference (without MC dropout)
    print("\nRunning standard inference...")
    model.enable_mc_dropout = False
    model.eval()

    # Process inputs
    image_inputs = processor(
        images=sample_images,
        # padding="max_length",
        # truncation=True,
        # max_length=77,
        size={"shortest_edge": 224, "longest_edge": 224},
        # do_resize=True,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=sample_prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    # Print input shapes for debugging
    print(f"Text input shape: {text_inputs['input_ids'].shape}")
    for key, value in image_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"Image input shape ({key}): {value.shape}")

    with torch.no_grad():
        # Run inference
        text_features = model.get_text_features(**text_inputs)
        print(f"Text features shape: {text_features.shape}")

        image_features = model.get_image_features(**image_inputs)
        print(f"Image features shape: {image_features.shape}")

        # Normalize features
        text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        image_features = image_features / torch.norm(
            image_features, dim=-1, keepdim=True
        )

        # Calculate scores and probabilities
        scores = model.logit_scale.exp() * (text_features @ image_features.T)[0]
        probs = torch.softmax(scores, dim=-1)

    print(f"Standard inference probabilities: {probs.cpu().numpy()}")

    # 8. MC Dropout inference
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


"""
    # 9. Visualize uncertainty (optional, if matplotlib is available)
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

    # 10. Test if MC dropout is actually applying stochasticity
    print("\nTesting if MC dropout is applying stochasticity...")
    model.enable_mc_dropout = True
    model.eval()

    with torch.no_grad():
        # Get features twice
        text_features1 = model.get_text_features(**text_inputs)
        text_features2 = model.get_text_features(**text_inputs)

        # Check if they're different (which they should be with MC dropout)
        is_different = not torch.allclose(text_features1, text_features2)

    print(f"MC dropout is working: {is_different}")

    # 11. Measure the average variance to quantify uncertainty
    if "text_features_samples" in mc_results:
        samples = mc_results["text_features_samples"]
        avg_variance = torch.mean(torch.var(samples, dim=0)).item()
        print(f"\nAverage variance across MC Dropout samples: {avg_variance:.6f}")
        print("Higher variance indicates more model uncertainty")
"""

if __name__ == "__main__":
    main()
