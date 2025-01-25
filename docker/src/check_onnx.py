import onnx
import onnxruntime as ort

# Load and check ONNX model
model_path = "solar_panel_segmentation.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)  # Validate model
print("ONNX model is valid!")

# Test with ONNX Runtime
session = ort.InferenceSession(model_path)
print("ONNX model loaded successfully!")
