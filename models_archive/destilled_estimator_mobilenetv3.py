import torch
import cv2
import torchvision.transforms as transforms
from models_archive.base_model import BaseEstimator
from objects_archive.student_object import StaticStudentMiVOLO 
import torch.nn.functional as F  

class MobileNetAgeEstimator(BaseEstimator):
    def __init__(self, weights_path, device, quant_type=None):
        self.device = device
        
        # 1. Instanciamos tu modelo destilado (MobileNetV3 Small 050)
        self.model = StaticStudentMiVOLO(num_outputs=3) 
        
        # 2. Carga de pesos destilados
        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            # Manejo del formato de tu checkpoint (dict con 'state_dict' o pesos directos)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Carga limpia de pesos
            self.model.load_state_dict(state_dict)
            print(f"[+] Pesos cargados en StaticStudentMiVOLO desde: {weights_path}")
        
        self.model.to(device)
        self.model.eval()

        # 3. Transformaciones (Deben coincidir con lo que usaste en run_destillation.py)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def estimate(self, face_crop, face_box=None):
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor_face = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            age_norm, gender_logits = self.model(tensor_face)
            
            # Revertir la normalización: (norm * rango) + avg
            # Asegúrate de usar los mismos valores que en run_destilation.py
            predicted_age = (age_norm.item() * 94.0) + 48.0 
            
            probs = F.softmax(gender_logits, dim=1)
            gender_idx = torch.argmax(probs, dim=1).item()
            return max(0.0, predicted_age), ("male" if gender_idx == 0 else "female"), probs[0, gender_idx].item()