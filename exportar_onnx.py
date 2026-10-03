import torch
import sys
import os

# 1. Route configuration
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mivolo.model.mi_volo import MiVOLO

weights_path = "weights/model_imdb_age_gender_4.22.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Loading MiVOLO from {weights_path}...")
# Cargamos el wrapper
mivolo_wrapper = MiVOLO(weights_path, device=str(device), half=False)
model = mivolo_wrapper.model.to(device).eval()

# 2. Esential Warm-up
# This initializes all internal buffers (including the ones that were failing).
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    print("[*] Performing warm-up inference...")
    _ = model(dummy_input)

print("[*] Exporting to ONNX via tracing...")

# 3. Exporting using Tracing (avoids jit.script, which causes the attribute error)
# We do not use scripted_model; we use the 'model' object directly.
try:
    torch.onnx.export(
        model, 
        dummy_input, 
        "mivolo_model.onnx",
        export_params=True,
        opset_version=18, 
        dynamo = True,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output']
    )
    print("[+] Success. File 'mivolo_model.onnx' generated")
except Exception as e:
    print(f"[!] Critical error during export: {e}")
    # If it fails due to an specific layer, it means the model is incompatible with ONNX.