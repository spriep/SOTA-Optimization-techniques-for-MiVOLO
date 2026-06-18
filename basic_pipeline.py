import cv2
import torch
import os
from collections import deque
import numpy as np


# Importing our modular models (detectors and estimators)
from models_archive.detector_retinaface_onnx import RetinaFaceDetector

from models_archive.estimator_resnet50 import ResNetAgeEstimator
from models_archive.estimator_mobilenetv3 import MobileNetAgeEstimator
from models_archive.estimator_mivolo_faceonly import MiVOLOAgeEstimator

# =====================================================================
# CONFIGURATION OF THE EXPERIMENT (device + model choices)
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "face": {
        # "path": "Pytorch_Retinaface/weights/mobilenet0.25_Final.pth",
        "path": "weights/retinaface_mnet.onnx",
        "class": RetinaFaceDetector
    },
    "age": {
        # "path": "weights/model_imdb_cross_person_4.22_99.46.pth.tar", #for official test with face+body (results prcessed_data/resultado_100_ofoocial.jpg)
        "path": "weights/model_only_age_imdb_4.29.pth.tar",
        "class": MiVOLOAgeEstimator
        #"path":  "weights/mobilenet_v3_ordinal_age.pth", # If we have pretrained weights its path should be written here
        #"class": MobileNetAgeEstimator
    }
}

'''
"age": {
    "path": estimator = "weights/mobilenet_v3_ordinal_age.pth", # If we have pretrained weights its path should be written here
    ""class"": MobileNetAgeEstimator
)'''

def print_pipeline_status(config, device):
    print("\n" + "="*40)
    print(f"BASIC PIPELINE INITIALIZED ON: {device.type.upper()}")
    print("-"*40)
    for model_name, info in config.items():
        print(f"[{model_name.upper():<6}] {info['class'].__name__:<20}")
    print("="*40 + "\n")

detector = MODEL_CONFIG["face"]["class"](onnx_path=MODEL_CONFIG["face"]["path"], device=device)
estimator = MODEL_CONFIG["age"]["class"](weights_path=MODEL_CONFIG["age"]["path"], device=device)

print_pipeline_status(MODEL_CONFIG, device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_video_path = os.path.join(BASE_DIR, "example_data", "video_test_2.mp4")
output_video_path = os.path.join(BASE_DIR, "example_data/processed_data", "new_modular_basic.mp4")


# =====================================================================
# MAIN PIPELINE FOR VIDEO PROCESSING
# =====================================================================
# OpenCV goes to the video on the harddrive, decodes 1 frame and extracts a NumPy matrix (three chanel photo), saves it in the frame variable, 
# this frame vraiable is procesed by RetinaFace (searches for faces and retourns the coordinates of the bounding boxes), then we crop the face and we send it to the Age ResNet, 
# which returns a number (the age estimation), OpenCV draws the green box and the text directly over that photo, and finally we save the frame in a new video file.
def run_pipeline(input_video_path, output_video_path, save_video=False):
    cap = cv2.VideoCapture(input_video_path) # Open the video file for reading
    if not cap.isOpened():
        print(f"[Error] Video couldnt be opened: {input_video_path}")
        exit()

    # Extraction of metadata from the video to configure the output video writer
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    out_video= None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))


    frame_count = 0
    age_estimation_per_frame = []
    window_size = 15
    age_buffer = deque(maxlen=window_size)

    while cap.isOpened():           # loop for processing each freame till the end of the video
        ret, frame = cap.read()     # frame = NumPy matrix, ret= boolean that indicates if the frame was read successfully or if we reached the end of the video
        if not ret:
            break
        frame_count += 1
        # Detector is invoqued  
        detections = detector.detect(frame) #returns a list where each detection is a list of 5 elements: [x_min, y_min, x_max, y_max, confidence_score]
        for det in detections:
            x1_f, y1_f, x2_f, y2_f, score = det
            
            # Casting to int to avoid errors
            x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)
            
            # Making sure the coordinates are within the frame --> avoid errors of drawing outside the limits
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Age estimation is invoqued
            '''age = int(round(estimator.estimate(frame, [x1, y1, x2, y2])))
            age_estimation_per_frame.append(age)'''
            raw_age = estimator.estimate(frame, [x1, y1, x2, y2])
            age_buffer.append(raw_age)
            if len(age_buffer) > 0:
                    smoothed_age = int(round(np.mean(age_buffer)))
                    age_estimation_per_frame.append(smoothed_age)
            # Drawing the results on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f"Age: {age}", (x1, y1 - 10), 
            cv2.putText(frame, f"Age: {smoothed_age}", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if save_video and out_video is not None:
            out_video.write(frame)

        if frame_count % 30 == 0:
            print(f" -> Processed frames: {frame_count}")
        

    cap.release()
    if save_video and out_video is not None:
        out_video.release()
        print(f"[+] Experiment successfully completed. Video available in: {output_video_path}")
    else:
        print(f"[+] Video successfully processed.")
    return age_estimation_per_frame, frame_count

if __name__ == "__main__":
    run_pipeline(input_video_path, output_video_path, save_video=True)