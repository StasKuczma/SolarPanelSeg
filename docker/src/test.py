import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image

# Paths
model_path = "solar_panel_segmentation.onnx"  # Path to the ONNX model
test_images_dir = "/workspace/data/test/"  # Path to test images
output_dir = "/workspace/data/test_predictions/"  # Path to save predictions

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Image preprocessing function
def preprocess_image(image_path, input_size=(256, 256)):
    """
    Preprocess the input image for the model.
    
    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Size to resize the image (height, width).
    
    Returns:
        (np.ndarray, np.ndarray): Preprocessed image and original image.
    """
    # Load the image
    original_image = Image.open(image_path).convert("RGB")
    
    # Resize and normalize
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    preprocessed_image = transform(original_image).unsqueeze(0)  # Add batch dimension
    
    return preprocessed_image.numpy(), np.array(original_image)

# Load ONNX model
def load_onnx_model(model_path):
    """
    Load the ONNX model for inference.
    
    Args:
        model_path (str): Path to the ONNX model.
    
    Returns:
        onnxruntime.InferenceSession: ONNX Runtime session.
    """
    return ort.InferenceSession(model_path)

# Postprocessing function
def postprocess_mask(mask, original_shape):
    """
    Postprocess the output mask to the original image size.
    
    Args:
        mask (np.ndarray): Model output mask.
        original_shape (tuple): Shape of the original image (height, width).
    
    Returns:
        np.ndarray: Resized mask.
    """
    mask = mask.squeeze()  # Remove batch and channel dimensions
    mask = (mask > 0.5).astype(np.uint8)  # Apply threshold
    mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]))  # Resize to original shape
    return mask_resized

# Perform inference on test images
def run_inference(test_images_dir, model, output_dir):
    """
    Run inference on test images and save predictions.
    
    Args:
        test_images_dir (str): Directory containing test images.
        model (onnxruntime.InferenceSession): ONNX model session.
        output_dir (str): Directory to save predicted masks.
    """
    # Get all test image paths
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(".jpg")]
    print(f"Found {len(test_images)} test images.")

    for image_name in test_images:
        image_path = os.path.join(test_images_dir, image_name)

        # Preprocess image
        input_image, original_image = preprocess_image(image_path)
        original_shape = original_image.shape[:2]  # Height, Width

        # Run model inference
        inputs = {model.get_inputs()[0].name: input_image}
        outputs = model.run(None, inputs)
        predicted_mask = outputs[0]  # Get the model output

        # Postprocess the mask
        final_mask = postprocess_mask(predicted_mask, original_shape)

        # Save the mask
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_mask.png")
        cv2.imwrite(output_path, final_mask * 255)  # Save mask as a binary image
        print(f"Saved mask for {image_name} at {output_path}")

# Main script
if __name__ == "__main__":
    # Load ONNX model
    onnx_model = load_onnx_model(model_path)
    print("ONNX model loaded successfully.")

    # Run inference on test images
    run_inference(test_images_dir, onnx_model, output_dir)
    print("Inference completed. Masks saved in:", output_dir)

