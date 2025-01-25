import torch
import segmentation_models_pytorch as smp

def export_to_onnx(model_path, output_path, input_size=(3, 512, 512), device="cuda"):
    """
    Exports a trained PyTorch model to ONNX format.

    Args:
        model_path (str): Path to the trained model (.pth file).
        output_path (str): Path to save the ONNX file.
        input_size (tuple): Input size for the model (channels, height, width).
        device (str): Device to load the model on ('cuda' or 'cpu').
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = smp.Unet(
        encoder_name="resnet34",        # Ensure this matches the encoder used during training
        encoder_weights=None,          # No pretrained weights; load your trained weights
        in_channels=3,                 # Input channels (e.g., RGB)
        classes=1                      # Number of output classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create a dummy input for ONNX export
    dummy_input = torch.randn(1, *input_size, dtype=torch.float32).to(device)  # Batch size 1, RGB image

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,            # Store trained parameters
        opset_version=11,              # ONNX opset version
        do_constant_folding=True,      # Optimize constant folding for inference
        input_names=["input"],         # Name of the input tensor
        output_names=["output"],       # Name of the output tensor
        dynamic_axes={                 # Allow dynamic axes for batch size and spatial dimensions
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )

    print(f"Model exported to ONNX format at: {output_path}")

# Define paths
model_path = "best_model.pth"  # Path to the trained model
output_path = "only_our_data.onnx"  # Path to save the ONNX file

# Export the model
export_to_onnx(model_path, output_path)

