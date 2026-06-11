import os
import sys
import torch

# 1. Conecting with the RetinaFace repository to import the model and its configuration
current_dir = os.getcwd()
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pytorch_Retinaface'))
sys.path.append(repo_dir)

from models.retinaface import RetinaFace
from data import cfg_mnet

# 2. Configuration
weights_path = "Pytorch_Retinaface/weights/mobilenet0.25_Final.pth" # weights we want to tranform to ONNX
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load the original model in PyTorch
os.chdir(repo_dir)
net = RetinaFace(cfg=cfg_mnet, phase='test')
os.chdir(current_dir)

# 4. Loading and cleaning the weights (as we did in the retinaface pytorch module)
state_dict = torch.load(weights_path, map_location=device)  # with the architecture layers and its weights
from collections import OrderedDict
new_state_dict = OrderedDict()      # Ordered dictionary to store the cleaned weights
for k, v in state_dict.items():      # Clean up the weight dictionary in case it was trained in parallel (im using a RTX 3050)
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.to(device)                 # Upload the entire model to the GPU cores
net.eval()                     # Turn off training layers (such as Dropout or BatchNorm)

print("[*] Exporting to ONNX...")

# 5. Cretaing a "dummy" input image so ONNX understands the data shape
dummy_input = torch.randn(1, 3, 640, 640).to(device)

# 6. Conversion --> using dynamic axes to accept any video size
torch.onnx.export(
    net, dummy_input, "retinaface_mnet.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['loc', 'conf', 'landms'],
    dynamic_axes={
        'input': {2: 'height', 3: 'width'}, # hight and width can be changed at inference time
        'loc': {1: 'num_anchors'},
        'conf': {1: 'num_anchors'},
        'landms': {1: 'num_anchors'}
    }
)

print("[+] Success. You have a new file 'retinaface_mnet.onnx' in your folder.")