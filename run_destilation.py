import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob

from objects_archive.student_object import StaticStudentMiVOLO
from objects_archive.destilation_utils import DistillationLoss, UTKandMorph2Dataset
import inspect
print(inspect.signature(UTKandMorph2Dataset.__init__))


def run_destillation(dataset_path, json_path, logits_dir, device='cuda'):
    # 1. Preprocessing Configuration
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 3. Initialize Student
    student = StaticStudentMiVOLO(num_outputs=3).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    criterion = DistillationLoss( alpha=0.5)

    # 4. Prepare DataLoader
    image_list = glob.glob(os.path.join(dataset_path, "*.jpg")) + glob.glob(os.path.join(dataset_path, "*.png"))
    
    my_dataset = UTKandMorph2Dataset(
        image_paths=image_list, 
        json_data_path=json_path, 
        output_logits_dir=logits_dir, 
        transform=preprocess
    )
    
    dataloader = DataLoader(my_dataset, batch_size=16, shuffle=True)
    
    # 5. Training Loop
    student.train()
    print(f"[*] Starting training with {len(image_list)} images.")
    
    print("[*] Initiating Phase 1: Gender Training...")
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
            
            # Standard normalization for the student
            labels_norm = ((labels_age.float() - 48.0) / 94.0).view(-1)  # Normalizamos entre 0 y 1
            
            optimizer.zero_grad()
            
            # Student inference
            student_outputs = student(images) 
            _, student_gender = student_outputs

            # Loss calculation (Teacher-Bounded Loss)
            #loss = criterion(student_outputs, teacher_logits, labels_norm, labels_gender)
            #print(f"Shape de teacher_logits: {teacher_logits.shape}")
            #print(f"Shape de student_gender: {student_gender.shape}")
            loss = criterion.ce_loss(student_gender, labels_gender).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} completed. average loss: {epoch_loss/len(dataloader):.4f}")


    print("[*] Starting Phase 2 with increased LR...")
    for epoch in range(3,6):
        epoch_loss = 0.0
        for i, (images, labels_age, labels_gender, teacher_logits) in enumerate(dataloader):
            # 1. Preparing data
            images = images.to(device)
            labels_norm = ((labels_age.float().to(device) - 48.0) / 94.0).view(-1)
            labels_gender = labels_gender.to(device)
            current_batch_size = images.size(0)
            teacher_logits = teacher_logits.to(device)[:current_batch_size] 
            
            # 2. Forward pass
            optimizer.zero_grad()
            student_outputs = student(images)
            loss = criterion(student_outputs, teacher_logits, labels_norm, labels_gender)
            
            # 3. Backpropagation
            loss.backward()
            
            # 4. Gradient health check (FIXED)
            total_grad = 0
            for name, param in student.named_parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().sum().item()
            
            if total_grad == 0:
                print(f"ALERTA: ZERO gradient in batch {i}! The model is not learning.")
            
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss/len(dataloader):.4f}")


    checkpoint_dict = {
        'state_dict': student.state_dict(),
        'min_age': 1, 'max_age': 95, 'avg_age': 48.0
    }
    torch.save(checkpoint_dict, "student_mivolo_final_simple_vits_UTK.pth")
    print("[+] Destilación finalizada y modelo guardado.")
    return student

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\UTKFace\set_TRAIN"
    #DATA_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\MORPH2\set_TRAIN"

    JSON_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\inference_outcomes_and_benchmark_log\distillation.json"
    #JSON_PATH = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\inference_outcomes_and_benchmark_log\distillation2.json"

    LOGITS_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\teacher_logits_cache"
    #LOGITS_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\teacher_logits_cache2"
    
    run_destillation(DATA_PATH, JSON_PATH, LOGITS_DIR)