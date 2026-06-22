import cv2
import os
import torch
from models_archive.detector_retinaface_onnx import RetinaFaceDetector

# CONFIGURACIÓN
DATASET_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\CD2\Album2"       # Carpeta con tus fotos originales
OUTPUT_DIR = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\Processed_MORPH2"        # Carpeta donde se guardarán los recortes
ERRORS_DIR = R"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\NOT_Processed_MORPH2"
LOG_FILE = os.path.join(ERRORS_DIR, "log_errores.txt")

# Asegurar directorios
for folder in [OUTPUT_DIR, ERRORS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Inicializar detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = RetinaFaceDetector(onnx_path="weights/retinaface_mnet.onnx", device=device)

def log_error_manually(filename, reason):
    """Copia la imagen problemática usando open() y registra el error"""
    src_path = os.path.join(DATASET_DIR, filename)
    dst_path = os.path.join(ERRORS_DIR, filename)
    
    # Copia de archivo binario nativa sin shutil
    with open(src_path, 'rb') as f_src:
        with open(dst_path, 'wb') as f_dst:
            f_dst.write(f_src.read())
    
    # Registro del error
    with open(LOG_FILE, "a") as log:
        log.write(f"Archivo: {filename} | Motivo: {reason}\n")

def process_dataset():
    # Limpiar log anterior si existe
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    for filename in os.listdir(DATASET_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(DATASET_DIR, filename)
            frame = cv2.imread(img_path)
            
            if frame is None:
                log_error_manually(filename, "No se pudo leer la imagen con OpenCV")
                continue

            detections = detector.detect(frame)

            if len(detections) == 0:
                log_error_manually(filename, "No se detectaron caras")
            elif len(detections) > 1:
                log_error_manually(filename, f"Se detectaron {len(detections)} caras")
            else:
                # Procesamiento exitoso
                x1, y1, x2, y2, score = map(int, detections[0])
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), face_crop)
                else:
                    log_error_manually(filename, "Error crítico durante el recorte")

    print(f"[+] Proceso terminado.")
    print(f"[+] Fotos procesadas en: {OUTPUT_DIR}")
    print(f"[+] Fotos con errores en: {ERRORS_DIR}")

if __name__ == "__main__":
    process_dataset()