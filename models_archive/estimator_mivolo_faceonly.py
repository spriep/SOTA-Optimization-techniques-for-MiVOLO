import os
import sys
import torch
import cv2
import torchvision.transforms as transforms
from models_archive.base_model import BaseEstimator
from torchvision.transforms import InterpolationMode
from mivolo.data.misc import prepare_classification_images



# mivolo folder path configuration for importing the MiVOLO model 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importimg the MiVOLO model class from the mivolo folder
try:
    from mivolo.model.mi_volo import MiVOLO
except ImportError as e:
    print("[!] FATAL ERROR: 'mivolo' folder not found in the project root.")
    print(e)
    sys.exit(1)

class MiVOLOAgeEstimator(BaseEstimator):
    def __init__(self, weights_path, device):
        self.device = device

        print("[*] Extracting normalization variables from the file...")
        ckpt = torch.load(weights_path, map_location="cpu")
        self.min_age = float(ckpt.get('min_age', 1.0))
        self.max_age = float(ckpt.get('max_age', 95.0))
        self.avg_age = float(ckpt.get('avg_age', 48.0))

        print(f"[*] Initialising MiVOLO (Vision Transformer) on {str(self.device).upper()}...")
        # Initialize the official wrapper for MiVOLO (Constructs the architecture + load the weights + prepares the onlyface mode)
        # The class constructs the transformer and load the weights
        self.mivolo_wrapper = MiVOLO(
            weights_path, 
            device=str(self.device),
            half=False # False --> half precisión (FP16) ; True --> full precisión (FP32) ----------------------------------------------------
        ) 
        # Extracting the mathematical model for inference
        self.model = self.mivolo_wrapper.model
        self.model.eval()

        # Standard normalization of ImageNet for the Transformers 
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_square_crop_with_padding(self, frame, x_min, y_min, x_max, y_max, margin=1.3):
        h, w = frame.shape[:2]
        box_w, box_h = x_max - x_min, y_max - y_min
        cx, cy = x_min + box_w // 2, y_min + box_h // 2
        side = int(max(box_w, box_h) * margin)
        
        new_x1, new_y1 = cx - side // 2, cy - side // 2
        new_x2, new_y2 = cx + side // 2, cy + side // 2
        
        pad_top, pad_bottom = max(0, -new_y1), max(0, new_y2 - h)
        pad_left, pad_right = max(0, -new_x1), max(0, new_x2 - w)
        
        valid_x1, valid_y1 = max(0, new_x1), max(0, new_y1)
        valid_x2, valid_y2 = min(w, new_x2), min(h, new_y2)
        
        crop_valid = frame[valid_y1:valid_y2, valid_x1:valid_x2]
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            return cv2.copyMakeBorder(crop_valid, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return crop_valid
    
    def estimate(self, frame, face_box):
        # 1. Recorte simple (sin resize ni padding complejo, solo el crop del array)
        def crop_img(f, box):
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = f.shape[:2]
            # Asegurar límites dentro del frame
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            return f[y1:y2, x1:x2]

        face_crop = crop_img(frame, face_box)

        if face_crop.size == 0:
            return None

        # 2. Usar las herramientas nativas de MiVOLO para preparar la imagen
        # Obtenemos mean/std del wrapper que inicializaste en __init__
        mean = self.mivolo_wrapper.data_config["mean"]
        std = self.mivolo_wrapper.data_config["std"]
        input_size = self.mivolo_wrapper.input_size # Esto lee el tamaño del checkpoint

        # prepare_classification_images hace el resize (bicubic), normalización yToTensor
        tensor_face = prepare_classification_images([face_crop], input_size, mean, std, device=self.device)
        
        # 3. Inferencia
        with torch.no_grad():
            # MiVOLO concatena la cara y el cuerpo en la dimensión 1 (canales)
            output = self.model(tensor_face)
            
            # Post-procesamiento igual que antes
            raw_age = output[0, 2].item() if output.shape[1] > 1 else output[0, 0].item()
            predicted_age = raw_age * (self.max_age - self.min_age) + self.avg_age
            return round(max(0.0, predicted_age), 1)

