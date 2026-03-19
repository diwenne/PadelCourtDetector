import torch
from tracknet import BallTrackerNet
import os

model_path = 'exps/padel_v3/model_best.pt'
output_onnx = 'exps/padel_v3/model_best.onnx'

print("Loading padel_v3 model for JIT tracing...")
model = BallTrackerNet(out_channels=6)
ckpt = torch.load(model_path, map_location='cpu')

if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.eval()

dummy_input = torch.randn(1, 3, 544, 960)

print("Exporting with torch.onnx.export...")
torch.onnx.export(
    model, 
    dummy_input, 
    output_onnx, 
    export_params=True, 
    opset_version=18, 
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output']
)

print("Export successful!")
if os.path.exists(output_onnx):
    print(f"File size: {os.path.getsize(output_onnx) / (1024*1024):.2f} MB")
