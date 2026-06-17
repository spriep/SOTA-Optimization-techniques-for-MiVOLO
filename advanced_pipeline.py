import cv2
import torch
import os
import time
from collections import deque
import numpy as np

from objects_archive.base_person import Person

from models_archive.detector_RFdetr import RFDETRDetector               # Body detector --> body crops + coordinates
from models_archive.detector_RTdetr import RTDETRDetector               
from models_archive.detector_retinaface_onnx import RetinaFaceDetector  # Face detector inside body crops --> face crops + coordinates
from models_archive.estimator_mivolo_facebody import MiVOLOAgeEstimator          # Age estimator face+body

user_home = os.path.expanduser("~") 
body_weights_path = os.path.join(user_home, ".roboflow", "models", "rf-detr-medium.pth")

MODEL_CONFIG = {
    "body": {
        #"path": body_weights_path,
        #"class": RFDETRDetector
        "path": "PekingU/rtdetr_r50vd",
        "class": RTDETRDetector
    },
    "face": {
        "path": "weights/retinaface_mnet.onnx",
        "class": RetinaFaceDetector
    },
    "age": {
        "path": "weights/model_imdb_cross_person_4.22_99.46.pth.tar",
        "class": MiVOLOAgeEstimator
    }
}
def print_pipeline_status(config, device):
    print("\n" + "="*40)
    print(f"ADVANCED PIPELINE INITIALIZED ON: {device.type.upper()}")
    print("-"*40)
    for model_name, info in config.items():
        # Obtenemos solo el nombre de la clase para que sea legible
        class_name = info['class'].__name__
        print(f"[{model_name.upper():<6}] {class_name:<20} | Path: {os.path.basename(info['path'])}")
    print("="*40 + "\n")

#EXPERIMENT CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_pipeline_status(MODEL_CONFIG, device)

# We instantiate the three models
body_detector = MODEL_CONFIG["body"]["class"](weights_path=MODEL_CONFIG["body"]["path"], device=device)
face_detector = MODEL_CONFIG["face"]["class"](onnx_path=MODEL_CONFIG["face"]["path"], device=device)
age_estimator = MODEL_CONFIG["age"]["class"](weights_path=MODEL_CONFIG["age"]["path"], device=device)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(BASE_DIR, "example_data", "video_test_2.mp4")
output_video_path = os.path.join(BASE_DIR, "example_data/processed_data", "output_RT.mp4")

# =====================================================================
# MAIN PIPELINE FOR VIDEO PROCESSING
# =====================================================================
def run_pipeline(input_video_path, output_video_path, save_video=False):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[Error] Video couldnt be opened: {input_video_path}")
        exit()

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    out_video = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))

    frame_count = 0
    age_estimation_per_frame=[]
            
    window_size = 15
    age_buffer = deque(maxlen=window_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        margin = 20
        retina_target_size = (320, 320)

        # --- 1) BODY DETECTION ---
        body_detections = body_detector.detect(frame)
        persons = [Person(id=i, body_box=[int(b[0]), int(b[1]), int(b[2]), int(b[3])], body_score=b[4]) for i, b in enumerate(body_detections)]

        # --- 2) FACE DETECTION (on body crops)---
        for person in persons:
            bx1, by1, bx2, by2 = person.body_box
            
            # 1. Add a safety margin in case one side is cut.
            y1, y2 = max(0, by1 - margin), min(h, by2 + margin)
            x1, x2 = max(0, bx1 - margin), min(w, bx2 + margin)
            body_crop_retina = frame[y1:y2, x1:x2]
            
            # 2. Resize for RetinaFace (avoids dimensional errors)
            body_crop_resized = cv2.resize(body_crop_retina, retina_target_size)
            
            # 3. Local Detection
            local_faces = face_detector.detect(body_crop_resized)
            local_faces = sorted(local_faces, key=lambda x: x[4], reverse=True)
            
            if local_faces is not None and len(local_faces) > 0:
                # Get the face with the best score
                fx1, fy1, fx2, fy2, f_score = local_faces[0]
                
                # Rescale face coordinates (return to the original cropped size)
                scale_x = (x2 - x1) / retina_target_size[0]
                scale_y = (y2 - y1) / retina_target_size[1]
                
                # Convert to global coordinates of the original frame
                person.face_box = [int(x1 + fx1 * scale_x), int(y1 + fy1 * scale_y), int(x1 + fx2 * scale_x), int(y1 + fy2 * scale_y)]
                person.face_score = f_score

        # --- 4) AGE ESTIMATION AND DRAWING AREA ---
            # Drawing the body square in blue
            if person.body_box:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID:{person.id}", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # If the person has his face visible we calculate the age estimation
            if person.has_face():   
                '''age = int(round(age_estimator.estimate(face_crop_mivolo, body_crop_mivolo)))
                age_estimation_per_frame.append(age)'''
                raw_age = age_estimator.estimate(frame, person.face_box, person.body_box)
                age_buffer.append(raw_age)
                if len(age_buffer) > 0:
                    smoothed_age = int(round(np.mean(age_buffer)))
                    age_estimation_per_frame.append(smoothed_age)

                cv2.rectangle(frame, (person.face_box[0], person.face_box[1]), (person.face_box[2], person.face_box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {smoothed_age}", (person.face_box[0], person.face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(frame, f"Age: {age}", (person.face_box[0], person.face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_video and out_video is not None:
            output_dir = os.path.dirname(output_video_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out_video.write(frame)
        
        if frame_count % 30 == 0:
            print(f" -> Processed frames: {frame_count}")

    cap.release()
    if save_video and out_video is not None:
        out_video.release()
        print(f"[+] Video successfully processed. Video available in: {output_video_path}")
    else:
        print(f"[+] Video successfully processed.")
    return age_estimation_per_frame, frame_count 

if __name__ == "__main__":
    age_estimations, frame_count = run_pipeline(input_video_path, output_video_path, save_video=True)
    print(age_estimations)