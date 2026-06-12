import cv2
import argparse
import torch

# ====================================================================
# PARCHE DE SEGURIDAD PARA PYTORCH 2.6+ 
# Sobrescribimos torch.load temporalmente para apagar el filtro weights_only
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load
# ====================================================================

# Ahora sí, importamos el Predictor
try:
    from mivolo.predictor import Predictor
except ImportError:
    from mivolo.predictor import Predictor

def main():
    print("[*] Iniciando el Pipeline 100% Oficial de MiVOLO...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/model_imdb_cross_person_4.22_99.46.pth.tar")
    # Apuntamos al YOLO modificado de MiVOLO
    parser.add_argument("--detector_weights", default="weights/yolov8x_person_face.pt") 
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--with_persons", action="store_true", default=False)
    parser.add_argument("--disable_faces", action="store_true", default=False)
    parser.add_argument("--draw", action="store_true", default=True)
    args = parser.parse_args([])

    predictor = Predictor(args, verbose=True)
    
    # Pon aquí la ruta a la foto del niño
    ruta_foto = "data/test_photo.jpg" 
    img = cv2.imread(ruta_foto)
    
    if img is None:
        print(f"[!] Error: No se encuentra la foto en {ruta_foto}")
        return
        
    print(f"[*] Procesando la foto con la IA de los autores...")
    preds, out_img = predictor.recognize(img)
    
    # Guardamos la foto pintada por el código oficial
    cv2.imwrite("processed_data/resultado_100_oficial.jpg", out_img)
    print("\n[+] Prueba completada. Abre 'processed_data/resultado_100_oficial.jpg' y mira la edad.")

if __name__ == "__main__":
    main()