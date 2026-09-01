import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob

# Importaciones de tus módulos
from objects_archive.student_object import StaticStudentMiVOLO
from objects_archive.destilation_utils import DistillationLoss, Morph2Dataset
import inspect
print(inspect.signature(Morph2Dataset.__init__))


def run_destillation(dataset_path, json_path, logits_dir, device='cuda'):
    # 1. Configuración de Preprocesamiento
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 3. Inicializar Estudiante
    student = StaticStudentMiVOLO(num_outputs=3).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    criterion = DistillationLoss( alpha=0.5)

    # 4. Preparar DataLoader
    image_list = glob.glob(os.path.join(dataset_path, "*.jpg")) + glob.glob(os.path.join(dataset_path, "*.png"))
    
    my_dataset = Morph2Dataset(
        image_paths=image_list, 
        json_data_path=json_path, 
        output_logits_dir=logits_dir, 
        transform=preprocess
    )
    
    dataloader = DataLoader(my_dataset, batch_size=16, shuffle=True)
    
    # 5. Bucle de Entrenamiento
    student.train()
    print(f"[*] Iniciando entrenamiento con {len(image_list)} imágenes.")
    
    print("[*] Iniciando Fase 1: Entrenamiento de Género...")
    for epoch in range(1,3):
        epoch_loss = 0.0 
        avg_age = 48.0 
        max_age = 95.0 
        min_age = 1.0
        for i, (images, labels_age, labels_gender, teacher_logits) in enumerate(dataloader):
            images = images.to(device)
            labels_age = labels_age.to(device)
            labels_gender = labels_gender.to(device)
            teacher_logits = teacher_logits.to(device).squeeze(1)
            
            # Normalización estándar para el estudiante
            labels_norm = ((labels_age.float() - 48.0) / 94.0).view(-1)  # Normalizamos entre 0 y 1
            
            optimizer.zero_grad()
            
            # Inferencia del estudiante
            student_outputs = student(images) 
            _, student_gender = student_outputs

            # Cálculo de pérdida (Teacher Bounded Loss)
            #loss = criterion(student_outputs, teacher_logits, labels_norm, labels_gender)
            #print(f"Shape de teacher_logits: {teacher_logits.shape}")
            #print(f"Shape de student_gender: {student_gender.shape}")
            loss = criterion.ce_loss(student_gender, labels_gender).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} completada. Loss promedio: {epoch_loss/len(dataloader):.4f}")


    print("[*] Iniciando Fase 2 con LR aumentado...")
    for epoch in range(3,6):
        epoch_loss = 0.0
        for i, (images, labels_age, labels_gender, teacher_logits) in enumerate(dataloader):
            # 1. Preparar datos
            images = images.to(device)
            labels_norm = ((labels_age.float().to(device) - 48.0) / 94.0).view(-1)
            labels_gender = labels_gender.to(device)
            current_batch_size = images.size(0)
            teacher_logits = teacher_logits.to(device)[:current_batch_size] 
            
            # 2. Paso hacia adelante
            optimizer.zero_grad()
            student_outputs = student(images)
            loss = criterion(student_outputs, teacher_logits, labels_norm, labels_gender)
            
            # 3. Retropropagación
            loss.backward()
            
            # 4. Chequeo de salud de gradientes (CORREGIDO)
            total_grad = 0
            for name, param in student.named_parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().sum().item()
            
            if total_grad == 0:
                print(f"ALERTA: ¡Gradiente CERO en batch {i}! El modelo no está aprendiendo.")
            
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} completada. Loss promedio: {epoch_loss/len(dataloader):.4f}")




    # 2. Crear un diccionario NUEVO y LIMPIO
    # Esto asegura que no arrastramos ninguna estructura anterior corrupta
    checkpoint_dict = {
        'state_dict': student.state_dict(),
        'min_age': 1, 'max_age': 95, 'avg_age': 48.0
    }
    torch.save(checkpoint_dict, "student_mivolo_final_mobile_large_100.pth")
    print("[+] Destilación finalizada y modelo guardado.")
    return student

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\UTKFace\set_TRAIN"
    JSON_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\inference_outcomes_and_benchmark_log\distillation.json"
    LOGITS_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\teacher_logits_cache"
    
    run_destillation(DATA_PATH, JSON_PATH, LOGITS_DIR)