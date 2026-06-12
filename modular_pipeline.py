import cv2
import torch
import os
import time 

# Importing our modular models (detectors and estimators)
from models_archive.detector_retinaface import RetinaFaceDetector
from models_archive.detector_retinaface_onnx import RetinaFaceDetector

from models_archive.estimator_resnet50 import ResNetAgeEstimator
from models_archive.estimator_mobilenetv3 import MobileNetAgeEstimator
from models_archive.estimator_mivolo import MiVOLOAgeEstimator

# =====================================================================
# CONFIGURATION OF THE EXPERIMENT (device + model choices)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciating the chosen detector and estimator for this experiment
detector = RetinaFaceDetector(
    #weights_path="Pytorch_Retinaface/weights/mobilenet0.25_Final.pth", 
    onnx_path="weights/retinaface_mnet.onnx",
    device=device
)

estimator = MiVOLOAgeEstimator( 
    #weights_path="weights/model_imdb_cross_person_4.22_99.46.pth.tar", #for official test with face+body (results prcessed_data/resultado_100_ofoocial.jpg)
    weights_path="weights/model_only_age_imdb_4.29.pth.tar",
    device=device
)

'''estimator = MobileNetAgeEstimator(
    weights_path="weights/mobilenet_v3_ordinal_age.pth", # If we have pretrained weights its path should be written here
    device=device
)'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_video_path = os.path.join(BASE_DIR, "data", "video_test_1.mp4")
output_video_path = os.path.join(BASE_DIR, "processed_data", "output_video_1_mivolo.mp4")

# ADAPTATION OF CROPS TO MIVOLO FORMAT (SQUARE WITH MARGIN)
def get_square_crop_with_padding(frame, x_min, y_min, x_max, y_max, margin=1.3):
    h, w = frame.shape[:2]
    
    # Dimensions of the original box
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    # Center of the face
    cx = x_min + box_w // 2
    cy = y_min + box_h // 2
    
    # Side of the new square with the margin
    side = int(max(box_w, box_h) * margin)
    
    # Ideal coordinates (they can go outside the image with negative numbers)
    new_x1 = cx - side // 2
    new_y1 = cy - side // 2
    new_x2 = cx + side // 2
    new_y2 = cy + side // 2
    
    # We calculate how many pixels are "missing" on each side if we go outside the border
    pad_top = max(0, -new_y1)
    pad_bottom = max(0, new_y2 - h)
    pad_left = max(0, -new_x1)
    pad_right = max(0, new_x2 - w)
    
    #  we cut only the valid part that exists inside the real frame
    valid_x1 = max(0, new_x1)
    valid_y1 = max(0, new_y1)
    valid_x2 = min(w, new_x2)
    valid_y2 = min(h, new_y2)
    
    crop_valid = frame[valid_y1:valid_y2, valid_x1:valid_x2]
    
    # If the box went outside the image on any side, fill the empty space with black to maintain the aspect ratio of the square without distorting the image
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        face_crop = cv2.copyMakeBorder(crop_valid, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        face_crop = crop_valid
        
    return face_crop

# =====================================================================
# MAIN PIPELINE FOR VIDEO PROCESSING
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

print(f"[*] Starting pipeline on: {device}")

frame_count = 0
while cap.isOpened():           # loop for processing each freame till the end of the video
    #capturing cicle start time
    t_start = time.perf_counter()
    ret, frame = cap.read()     # frame = NumPy matrix, ret= boolean that indicates if the frame was read successfully or if we reached the end of the video
    if not ret:
        break
    frame_count += 1

    #first control point --> End of reading the physical frame
    t_read = time.perf_counter()
    # Detector is invoqued  
    detections = detector.detect(frame) #returns a list where each detection is a list of 5 elements: [x_min, y_min, x_max, y_max, confidence_score]
    #second control point --> End of the detection process
    t_detect = time.perf_counter()
    
    for det in detections:
        x_min, y_min, x_max, y_max, score = det
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        face_crop = get_square_crop_with_padding(frame, x_min, y_min, x_max, y_max, margin=1.3)

        if face_crop.size == 0:
            continue
            
        # Age estimation is invoqued
        predicted_age_float = estimator.estimate(face_crop)
        predicted_age_int = int(round(predicted_age_float))
        
        # Drawing the results on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {predicted_age_int}", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Third control point --> End of the estimation process
    t_estimate = time.perf_counter()

    out_video.write(frame)
    if frame_count % 30 == 0:
        print(f" -> Processed frames: {frame_count}")
    
    #fourth control point --> End of the cycle
    t_write = time.perf_counter()
    '''time_read_ms     = (t_read - t_start) * 1000
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