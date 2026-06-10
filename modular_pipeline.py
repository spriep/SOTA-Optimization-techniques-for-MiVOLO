# pipeline_modular.py
import cv2
import torch
import os
import time 

# Importamos nuestras piezas de Lego moleculares
from models_archive.detector_retinaface import RetinaFaceDetector
from models_archive.estimator_resnet50 import ResNetAgeEstimator

# =====================================================================
# CONFIGURATION OF THE EXPERIMENT (device + model choices)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciamos el detector y estimador elegidos para este experimento
detector = RetinaFaceDetector(
    weights_path="Pytorch_Retinaface/weights/mobilenet0.25_Final.pth", 
    device=device
)

estimator = ResNetAgeEstimator(
    weights_path=None, # Pon aquí la ruta a tus pesos entrenados cuando los tengas
    device=device
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_video_path = os.path.join(BASE_DIR, "data", "video_test_1.mp4")
output_video_path = os.path.join(BASE_DIR, "processed_data", "output_video_1.mp4")

# =====================================================================
# PRINCIPAL PIPELINE FOR VIDEO PROCESSING
# =====================================================================
# OpenCV goes to the video on the harddrive, decodes 1 frame and extracts a NumPy matrix (three chanel photo), saves it in the frame variable, 
# this frame vraiable is procesed by RetinaFace (searches for faces and retourns the coordinates of the bounding boxes), then we crop the face and we send it to the Age ResNet, 
# which returns a number (the age estimation), OpenCV draws the green box and the text directly over that photo, and finally we save the frame in a new video file.
cap = cv2.VideoCapture(input_video_path) # Open the video file for reading
if not cap.isOpened():
    print(f"[Error] No se pudo abrir el vídeo: {input_video_path}")
    exit()

# Extraction of metadata from the video to configure the output video writer
frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps      = cap.get(cv2.CAP_PROP_FPS)
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))

print(f"[*] Iniciando experimento modular en: {device}")

frame_count = 0
while cap.isOpened():           # loop for processing each freame till the end of the video
    '''t_start = time.perf_counter()'''
    ret, frame = cap.read()     # frame = NumPy matrix, ret= boolean that indicates if the frame was read successfully or if we reached the end of the video
    if not ret:
        break
    frame_count += 1

    '''t_read = time.perf_counter()'''
    # Detector is invoqued  
    detections = detector.detect(frame) #returns a list where each detection is a list of 5 elements: [x_min, y_min, x_max, y_max, confidence_score]
    '''t_detect = time.perf_counter()'''
    
    for det in detections:
        x_min, y_min, x_max, y_max, score = det
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_w, x_max), min(frame_h, y_max)
        
        face_crop = frame[y_min:y_max, x_min:x_max] # cutting the photo croped by the bounding box coordinates
        if face_crop.size == 0:
            continue
            
        # Age estimation is invoqued
        predicted_age_float = estimator.estimate(face_crop)
        predicted_age_int = int(round(predicted_age_float))
        
        # Drawing the results on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {predicted_age_int}", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    '''t_estimate = time.perf_counter()'''

    out_video.write(frame)
    if frame_count % 30 == 0:
        print(f" -> Processed frames: {frame_count}")
    
    '''t_write = time.perf_counter()
    time_read_ms     = (t_read - t_start) * 1000
    time_detect_ms   = (t_detect - t_read) * 1000
    time_estimate_ms = (t_estimate - t_detect) * 1000
    time_write_ms    = (t_write - t_estimate) * 1000
    
    total_time_ms    = (t_write - t_start) * 1000
    current_fps      = 1000 / total_time_ms if total_time_ms > 0 else 0

    # Imprimimos el reporte cada 30 frames para no colapsar la terminal
    if frame_count % 30 == 0:
        print(f"\n[*] Reporte de rendimiento - Frame {frame_count}:")
        print(f"  -> Lectura de vídeo: {time_read_ms:>6.1f} ms")
        print(f"  -> Modelo Detector:  {time_detect_ms:>6.1f} ms")
        print(f"  -> Modelo Estimador: {time_estimate_ms:>6.1f} ms")
        print(f"  -> Escritura disco:  {time_write_ms:>6.1f} ms")
        print(f"  => TIEMPO TOTAL:     {total_time_ms:>6.1f} ms | Velocidad: {current_fps:.1f} FPS")'''

cap.release()
out_video.release()
print(f"[+] Experiment successfully completed. Video available in: {output_video_path}")