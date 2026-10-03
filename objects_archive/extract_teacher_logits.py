import torch
import os
import json
import glob
from torchvision import transforms
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models_archive.estimator_mivolo_faceonly import MiVOLOAgeEstimator

# Configuration
#DATA_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\UTKFace\set_TRAIN"
DATA_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\MORPH2\set_TRAIN"

#JSON_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\inference_outcomes_and_benchmark_log\distillation.json" # Tu archivo JSON
JSON_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\inference_outcomes_and_benchmark_log\distillation2.json" # Tu archivo JSON

#OUTPUT_LOGITS_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\teacher_logits_cache"  # Carpeta donde se guardarán los logits del profesor
OUTPUT_LOGITS_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\teacher_logits_cache2"  # Carpeta donde se guardarán los logits del profesor

#TEACHER_PATH = r'C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\weights\model_imdb_age_gender_4.22.pth.tar'
TEACHER_PATH = r'C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\weights\model_imdb_age_gender_4.22.pth.tar'

os.makedirs(OUTPUT_LOGITS_DIR, exist_ok=True)
device = 'cuda'

# Load Profesor
teacher_wrapper = MiVOLOAgeEstimator(weights_path=TEACHER_PATH, device=device)
teacher_model = teacher_wrapper.model.to(device).eval()
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

with open(JSON_PATH, 'r') as f:
    metadata = json.load(f)

print("[*] Generating teacher logit cache...")
for filename in metadata.keys():
    # We assume the filename on disk ends with .chip.jpg.
    #img_name = filename.replace('.json', '') 
    #img_path = os.path.join(DATA_PATH, img_name)
    
    img_path = os.path.join(DATA_PATH, filename)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = teacher_model(tensor)
            
        #save_path = os.path.join(OUTPUT_LOGITS_DIR, img_name.replace('.jpg', '.pt'))
        base_name = filename.split('.jpg')[0] # Toma todo lo anterior al primer .jpg
        save_filename = f"{base_name}.pt"
        save_path = os.path.join(OUTPUT_LOGITS_DIR, save_filename)
        print(f"Logit saved to: {save_path}")
        
        
        torch.save(logits.cpu(), save_path)


print("[+] Caché generada.")