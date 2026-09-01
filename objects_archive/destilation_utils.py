import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
import json


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, device='cuda'):
        super().__init__()
        self.alpha = alpha
        # CAMBIO 1: reduction='none' es OBLIGATORIO para calcular pérdidas por instancia 
        self.mse_loss = nn.MSELoss(reduction='none') 
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, student_outputs, teacher_logits, labels_norm, labels_gender):
        student_age, student_gender = student_outputs
        
        # 1. Extracción precisa de Logits del Profesor
        # Edad: Valor 3 (índice 2)
        t_age_raw = teacher_logits[:, 2].view(-1, 1)
        # Género: Valores 1 y 2 (índices 0:2)
        t_gender_logits = teacher_logits[:, 0:2]
        
        # 2. Des-normalización: Pasamos de [0.38] a la edad real
        t_age_scaled = (t_age_raw * 94.0) + 48.0 
        
        # 3. Comparación (MSE y CE)
        y_age = labels_norm.view(-1, 1)
        
        # Error MSE individual (estudiante vs gt)
        mse_s = (student_age - y_age)**2
        # Error MSE individual (profesor vs gt)
        mse_t = (t_age_scaled - y_age)**2
        
        # Error CE individual (estudiante vs gt)
        ce_s = self.ce_loss(student_gender, labels_gender)
        # Error CE individual (profesor vs gt)
        ce_t = self.ce_loss(t_gender_logits, labels_gender)
        
        # 4. Máscaras (Teacher Bounded Loss) [cite: 240]
        # Destilamos solo si el profesor es más preciso [cite: 196, 240]
        mse_bounded = torch.where(mse_t < mse_s, (student_age - t_age_scaled)**2, torch.zeros_like(mse_s))
        ce_bounded = torch.where(ce_t < ce_s, self.ce_loss(student_gender, labels_gender), torch.zeros_like(ce_s))
        
        # 5. Pérdida Final [cite: 201]
        loss_age = ((1 - self.alpha) * mse_s.mean()) + (self.alpha * mse_bounded.mean())
        loss_gender = ((1 - self.alpha) * ce_s.mean()) + (self.alpha * ce_bounded.mean())
        
        return loss_age + loss_gender
    
class Morph2Dataset(Dataset):
    def __init__(self, image_paths, json_data_path, output_logits_dir, transform=None):
        self.image_paths = image_paths
        self.json_data_path = json_data_path
        self.output_logits_dir = output_logits_dir
        self.transform = transform
        with open(self.json_data_path, 'r') as f:
            self.metadata = json.load(f)
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        
        # 1. Cargar metadatos
        data = self.metadata.get(filename, {})
        label_age = torch.tensor(data.get('age', 0.0), dtype=torch.float32)
        label_gender = torch.tensor(0 if data.get('gender') == 'male' else 1, dtype=torch.long)
        
        # 2. Cargar logits
        logit_path = os.path.join(self.output_logits_dir, filename.replace('.jpg', '.pt'))
        teacher_logits = torch.load(logit_path)
        
        # --- CORRECCIÓN CRÍTICA ---
        # Si el archivo guardado tiene dimensión [1, 3] o [N, 3], 
        # nos aseguramos de devolver solo el primer elemento: [3]
        if teacher_logits.dim() > 1:
            teacher_logits = teacher_logits[0] 
        # ---------------------------
        
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label_age, label_gender, teacher_logits