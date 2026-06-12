import os
import sys
import torch
import cv2
import torchvision.transforms as transforms
from models_archive.base_model import BaseEstimator
from torchvision.transforms import InterpolationMode


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
    
    def estimate(self, face_crop, body_crop):
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        body_crop_rgb = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
        tensor_face = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)
        tensor_body = self.transform(body_crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Passing the fases through the transformer
            output = self.model(torch.cat([tensor_face, tensor_body], dim=1)) # Directly using the model for inference, without the wrapper's postprocessing (which is designed for the original MiVOLO outputs)
            
            # If the models outputs 1 number (age)
            if output.shape[1] == 1:
                raw_age = output[0, 0].item()
            # If the model outputs 3 numbers (Gender + Age), we take the third one
            else:
                raw_age = output[0, 2].item()
            
            # reverting normalization to get the real age estimation in years
            predicted_age = raw_age * (self.max_age - self.min_age) + self.avg_age
            predicted_age = max(0.0, predicted_age)
            
            
            return round(predicted_age, 1) # round to 1 decimal

