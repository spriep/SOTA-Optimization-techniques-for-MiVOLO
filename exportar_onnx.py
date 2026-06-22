import torch
import sys
import os

# 1. Configuración de rutas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from mivolo.model.mi_volo import MiVOLO

weights_path = "weights/model_imdb_age_gender_4.22.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Cargando MiVOLO desde {weights_path}...")
# Cargamos el wrapper
mivolo_wrapper = MiVOLO(weights_path, device=str(device), half=False)
model = mivolo_wrapper.model.to(device).eval()

# 2. Warm-up esencial
# Esto inicializa todos los buffers internos (incluyendo los que fallaban)
dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    print("[*] Realizando inferencia de warm-up...")
    _ = model(dummy_input)

print("[*] Exportando a ONNX mediante tracing...")

# 3. Exportación usando Tracing (evita jit.script que es el que causa el error de atributos)
# NO usamos scripted_model, usamos el objeto 'model' directamente.
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
    print("[+] Éxito. Archivo 'mivolo_model.onnx' generado.")
except Exception as e:
    print(f"[!] Error crítico durante la exportación: {e}")
    # Si sigue fallando por col2im, significa que el modelo es incompatible con ONNX