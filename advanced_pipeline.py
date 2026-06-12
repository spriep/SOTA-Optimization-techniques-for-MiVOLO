import cv2
import torch
import os
import time

from objects_archive.base_person import Person

from models_archive.detector_rtdetr import RTDETRDetector               # Body detector --> body crops + coordinates
from models_archive.detector_retinaface_onnx import RetinaFaceDetector  # Face detector inside body crops --> face crops + coordinates
from models_archive.estimator_mivolo_facebody import MiVOLOAgeEstimator          # Age estimator face+body

#EXPERIMENT CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Inizialating Advanced Pipeline on: {device}")

# We instantiate the three models
body_detector = RTDETRDetector(weights_path="medium", device=device)
face_detector = RetinaFaceDetector(onnx_path="weights/retinaface_mnet.onnx", device=device)
age_estimator = MiVOLOAgeEstimator(weights_path="weights/model_imdb_cross_person_4.22_99.46.pth.tar", device=device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(BASE_DIR, "data", "video_test_2.mp4")
output_video_path = os.path.join(BASE_DIR, "processed_data", "output_null.mp4")

#PADDING FOR MIVOLO
def get_square_crop_with_padding(frame, x_min, y_min, x_max, y_max, margin=1.3):
    h, w = frame.shape[:2]
    box_w, box_h = x_max - x_min, y_max - y_min
    cx, cy = x_min + box_w // 2, y_min + box_h // 2
    # Side of the new square with the margin
    side = int(max(box_w, box_h) * margin)
    
    new_x1, new_y1 = cx - side // 2, cy - side // 2
    new_x2, new_y2 = cx + side // 2, cy + side // 2
    
    pad_top, pad_bottom = max(0, -new_y1), max(0, new_y2 - h)
    pad_left, pad_right = max(0, -new_x1), max(0, new_x2 - w)
    
    valid_x1 = max(0, new_x1)
    valid_y1 = max(0, new_y1)
    valid_x2 = min(w, new_x2)
    valid_y2 = min(h, new_y2)
    
    crop_valid = frame[valid_y1:valid_y2, valid_x1:valid_x2]
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        face_crop = cv2.copyMakeBorder(crop_valid, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        face_crop = crop_valid
        
    return face_crop

# =====================================================================
# MAIN PIPELINE FOR VIDEO PROCESSING
# =====================================================================
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"[Error] Video couldnt be opened: {input_video_path}")
    exit()

frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps      = cap.get(cv2.CAP_PROP_FPS)
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # --- 1) BODY DETECTION ---
    body_detections = body_detector.detect(frame)
    persons = [Person(id=i, body_box=[int(b[0]), int(b[1]), int(b[2]), int(b[3])], body_score=b[4]) for i, b in enumerate(body_detections)]

    # --- 2) FACE DETECTION ---
    face_detections = face_detector.detect(frame)

    # --- 3) FACE+BODY MATCHING ---
    for fx1, fy1, fx2, fy2, f_score in face_detections:                     # faces center is calculated
        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2                       #if this position is inside someone bodybox, match, we save the metadata on the person object
        for person in persons:
            bx1, by1, bx2, by2 = person.body_box
            if bx1 <= fcx <= bx2 and by1 <= fcy <= by2:
                person.face_box = [int(fx1), int(fy1), int(fx2), int(fy2)]
                person.face_score = f_score
                break

    # --- 4) AGE ESTIMATION AND DRAWING AREA ---
    for person in persons:
        # 1. Dibujamos el cuerpo en azul
        if person.body_box:
            bx1, by1, bx2, by2 = person.body_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID:{person.id}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # If the person has his face visible we calculate the age estimation
        if person.has_face():
            fx1, fy1, fx2, fy2 = person.face_box
            
            # Crop format adpatation and enetering MiVOLO
            face_crop = get_square_crop_with_padding(frame, fx1, fy1, fx2, fy2, margin=1.3)
            body_crop = frame[person.body_box[1]:person.body_box[3], person.body_box[0]:person.body_box[2]]
            
            if face_crop.size > 0:
                face_crop = get_square_crop_with_padding(frame, *person.face_box, margin=1.3)
                age = int(round(age_estimator.estimate(face_crop, body_crop)))
                cv2.rectangle(frame, (person.face_box[0], person.face_box[1]), (person.face_box[2], person.face_box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age}", (person.face_box[0], person.face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_video.write(frame)
    if frame_count % 30 == 0:
        print(f" -> Processed frames: {frame_count}")

cap.release()
out_video.release()
print(f"[+] Experiment successfully completed. Video available in: {output_video_path}")