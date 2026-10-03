import torch
import os

def check_teacher_logits(logits_dir, num_files_to_check=5):
    files = [f for f in os.listdir(logits_dir) if f.endswith('.pt')][:num_files_to_check]
    
    print(f"[*] Inspeccionando {len(files)} archivos en {logits_dir}...\n")
    
    for filename in files:
        path = os.path.join(logits_dir, filename)
        logits = torch.load(path)
        
        # 1. Verificación de forma
        print(f"Archivo: {filename}")
        print(f"  Shape: {logits.shape}")
        
        # 2. Verificación de contenido (Asumiendo que el índice 2 es la edad)
        # Ajusta este índice si tu profesor guarda la edad en otra posición
        age_pred = logits[0, 2] if logits.dim() > 1 else logits[2]
        
        print(f"  Valor (Edad estimada): {age_pred.item():.4f}")
        
        # 3. Alerta de rangos imposibles
        if age_pred < 0 or age_pred > 100:
            print(f"  [!] ALERTA: Valor de edad fuera de rango humano (0-100).")
        
        print("-" * 30)

        print(f"  Valor 1: {logits[0, 0].item():.4f}")
        print(f"  Valor 2: {logits[0, 1].item():.4f}")
        print(f"  Valor 3: {logits[0, 2].item():.4f}")

if __name__ == "__main__":
    LOGITS_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\teacher_logits_cache"
    check_teacher_logits(LOGITS_DIR)