import cv2
import torch
import os
from models_archive.estimator_mivolo_faceonly import MiVOLOAgeEstimator

# Configuración
MODEL_CONFIG = {
    "age": {
        # "path": "weights/model_imdb_cross_person_4.22_99.46.pth.tar", #for official test with face+body (results prcessed_data/resultado_100_ofoocial.jpg)
        "path": "weights/model_only_age_imdb_4.29.pth.tar",
        "class": MiVOLOAgeEstimator
        #"path":  "weights/mobilenet_v3_ordinal_age.pth", # If we have pretrained weights its path should be written here
        #"class": MobileNetAgeEstimator
    }
}
def print_pipeline_status(config, device):
    print("\n" + "="*40)
    print(f"BASIC PIPELINE INITIALIZED ON: {device.type.upper()}")
    print("-"*40)
    for model_name, info in config.items():
        print(f"[{model_name.upper():<6}] {info['class'].__name__:<20}")
    print("="*40 + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
estimator = MODEL_CONFIG["age"]["class"](weights_path=MODEL_CONFIG["age"]["path"], device=device)

print_pipeline_status(MODEL_CONFIG, device)

def process_image(image_path):
    # 1. Leer la imagen
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] No se pudo cargar la imagen: {image_path}")
        return

    h, w = frame.shape[:2]
    
    # 2. Definir la "caja" como la imagen completa (x_min, y_min, x_max, y_max)
    # Si la imagen es una cara centrada, esto funcionará directamente
    face_box = [0, 0, w, h]
    
    # 3. Estimar la edad
    # El método estimate en tu clase actual recorta basándose en face_box
    predicted_age = estimator.estimate(frame, face_box)
    
    print(f"[*] Edad estimada: {predicted_age} años")
    
    # (Opcional) Guardar resultado visual
    cv2.putText(frame, f"Age: {predicted_age}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imwrite("resultado_estimacion.jpg", frame)
    print("[+] Imagen procesada guardada como 'resultado_estimacion.jpg'")

if __name__ == "__main__":
    process_image("ruta/a/tu/imagen.jpg")