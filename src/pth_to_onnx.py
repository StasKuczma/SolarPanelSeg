from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import segmentation_models_pytorch as smp


MODEL_PATH = "./models/best_model.pth"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

x = torch.rand([1, 3, 512, 512]) 

torch.onnx.export(model,
                x, 
                './models/best_model.onnx', 
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output'],
                do_constant_folding=False)


