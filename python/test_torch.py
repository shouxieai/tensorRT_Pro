
import torch
import torchvision.models as models
import trtpy as tp

device = "cuda" if torch.cuda.is_available() else "cpu"

input     = torch.full((5, 3, 224, 224), 0.3).to(device)
model     = models.resnet18(True).eval().to(device)
trt_model = tp.convert_torch_to_trt(model, input)
torch_out = model(input)
trt_out   = trt_model(input)

abs_diff = (torch_out - trt_out).abs().max()
print(f"Torch and TRTModel abs diff is {abs_diff}")