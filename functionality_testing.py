import torch
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from PIL import Image
import numpy as np
import os

def test_vanilla_clip_image_path():
    """Test the vanilla CLIP model image pathway directly from HuggingFace."""
    print("\n===== Testing Vanilla CLIP Image Pathway =====")
    
    # Set to True to see detailed config information
    VERBOSE = True
    
    # 1. Load the model directly from HuggingFace
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    try:
        print(f"Loading vanilla model from {model_name}")
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully")
        
        if VERBOSE:
            # Print model configuration details
            config = model.config
            print("\nModel Configuration:")
            print(f"- Vision embed dim: {config.vision_config.hidden_size}")
            print(f"- Vision image size: {config.vision_config.image_size}")
            print(f"- Vision patch size: {config.vision_config.patch_size}")
            print(f"- Projection dim: {config.projection_dim}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # 2. Create a simple test image
    image_size = config.vision_config.image_size  # Get the expected image size
    print(f"\nCreating test image with size {image_size}x{image_size}")
    
    try:
        # Create an image with the correct size from the model config
        test_image = Image.new('RGB', (image_size, image_size), color=(100, 150, 200))
        
        # 3. Process the image with the processor
        print("Processing image...")
        image_inputs = processor(
            images=test_image,
            return_tensors="pt"
        )
        
        # Print the input shapes
        print("\nProcessor outputs:")
        for key, tensor in image_inputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")
        
        # 4. Extract image features
        print("\nExtracting image features...")
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
        
        print(f"Image features shape: {image_features.shape}")
        print("✅ Successfully extracted image features from vanilla model")
        return True
        
    except Exception as e:
        print(f"❌ Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_clip_image_path():
    """Test the custom CLIP model's image pathway."""
    print("\n===== Testing Custom CLIP Model Image Pathway =====")
    
    try:
        # 1. Import your custom model
        from trainer.models.clip_model import CLIPModel, ClipModelConfig
        
        # 2. Create model with zero dropout
        print("Creating custom model...")
        model_config = ClipModelConfig(
            pretrained_model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            dropout_rate=0.0,  # No dropout for this test
            enable_mc_dropout=False
        )
        
        model = CLIPModel(model_config)
        model.eval()
        print("Custom model created successfully")
        
        # 3. Load processor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        # 4. Get the expected image size
        # Access the underlying model's config
        image_size = model.model.config.vision_config.image_size
        print(f"Expected image size from model config: {image_size}")
        
        # 5. Create a test image with the correct size
        test_image = Image.new('RGB', (image_size, image_size), color=(100, 150, 200))
        
        # 6. Process the image
        print("Processing image...")
        image_inputs = processor(
            images=test_image,
            return_tensors="pt"
        )
        
        # Print the input shapes
        print("\nProcessor outputs:")
        for key, tensor in image_inputs.items():
            if isinstance(tensor, torch.Tensor):
                print(f"- {key}: {tensor.shape}")
        
        # 7. Extract image features
        print("\nExtracting image features...")
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
        
        print(f"Image features shape: {image_features.shape}")
        print("✅ Successfully extracted image features from custom model")
        return True
        
    except Exception as e:
        print(f"❌ Error during custom model test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alternative_sizes():
    """Test different image sizes to find what works."""
    print("\n===== Testing Alternative Image Sizes =====")
    
    try:
        # Load vanilla model
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        model.eval()
        
        # Try different image sizes
        test_sizes = [224, 336, 448]
        
        for size in test_sizes:
            print(f"\nTesting image size: {size}x{size}")
            try:
                # Create image
                test_image = Image.new('RGB', (size, size), color=(100, 150, 200))
                
                # Process with explicit size
                image_inputs = processor(
                    images=test_image,
                    size={"height": size, "width": size},
                    return_tensors="pt"
                )
                
                # Extract features
                with torch.no_grad():
                    image_features = model.get_image_features(**image_inputs)
                
                print(f"✅ Success with size {size}x{size} - features shape: {image_features.shape}")
                
            except Exception as e:
                print(f"❌ Failed with size {size}x{size}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error in alternative size test: {e}")
        return False

def inspect_loaded_weights():
    """Inspect your loaded weights for the custom model."""
    print("\n===== Inspecting Loaded Weights =====")
    
    try:
        # Load the state dict
        weights_path = "outputs/checkpoint-gstep100/pytorch_model.bin"
        if not os.path.exists(weights_path):
            print(f"❌ Weights file not found at {weights_path}")
            return False
            
        state_dict = torch.load(weights_path, weights_only=False, map_location="cpu")
        
        # Count parameters and check for NaN values
        total_params = 0
        nan_params = 0
        
        print("\nAnalyzing weight parameters:")
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                param_count = tensor.numel()
                total_params += param_count
                
                # Check for NaNs
                nan_count = torch.isnan(tensor).sum().item()
                if nan_count > 0:
                    nan_params += nan_count
                    print(f"⚠️ Found {nan_count} NaN values in {key}")
        
        print(f"Total parameters: {total_params:,}")
        print(f"NaN parameters: {nan_params:,}")
        
        # Check if vision embedding layer exists
        key_exists = False
        for k in state_dict.keys():
            if 'model.vision_model.embeddings' in k or 'vision_model.embeddings' in k:
                key_exists = True
                print(f"✅ Found vision embedding layer: {k}")
                
        if not key_exists:
            print("⚠️ No vision embedding layer found in weights")
        
        return True
    
    except Exception as e:
        print(f"Error inspecting weights: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("==================================================")
    print("CLIP MODEL IMAGE PATHWAY DIAGNOSTIC TESTS")
    print("==================================================")
    
    # Run tests
    vanilla_test = test_vanilla_clip_image_path()
    custom_test = test_custom_clip_image_path()
    size_test = test_alternative_sizes()
    weights_test = inspect_loaded_weights()
    
    # Print overall results
    print("\n==================================================")
    print("TEST RESULTS SUMMARY")
    print("==================================================")
    print(f"Vanilla CLIP image pathway: {'✅ WORKING' if vanilla_test else '❌ FAILED'}")
    print(f"Custom CLIP image pathway: {'✅ WORKING' if custom_test else '❌ FAILED'}")
    print(f"Alternative image sizes: {'✅ TESTED' if size_test else '❌ FAILED'}")
    print(f"Weights inspection: {'✅ COMPLETED' if weights_test else '❌ FAILED'}")
    
    print("\nRECOMMENDATIONS:")
    if not vanilla_test:
        print("- The vanilla CLIP model is failing - this may indicate an installation issue")
    if vanilla_test and not custom_test:
        print("- The base model works but your custom model doesn't - check your CLIPModel implementation")
    if vanilla_test and custom_test:
        print("- Both models work with this test - the issue might be in your original inference code")
    if weights_test and 'nan_params' in locals() and nan_params > 0:
        print("- Your weights contain NaN values which could cause numerical issues")