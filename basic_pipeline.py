import cv2
import torch
import os

# Importing our modular models (detectors and estimators)
from models_archive.detector_retinaface import RetinaFaceDetector
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

input_video_path = os.path.join(BASE_DIR, "example_data", "video_test_1.mp4")
output_video_path = os.path.join(BASE_DIR, "example_data/processed_data", "output_video_1_mivolo.mp4")

# ADAPTATION OF CROPS TO MIVOLO FORMAT (SQUARE WITH MARGIN)
def get_square_crop_with_padding(frame, x_min, y_min, x_max, y_max, margin=1.3):
    h, w = frame.shape[:2]
    # Dimensions of the original box
    box_w, box_h = x_max - x_min, y_max - y_min
    # Center of the face
    cx, cy = x_min + box_w // 2, y_min + box_h // 2
    # Side of the new square with the margin
    side = int(max(box_w, box_h) * margin)
    # Ideal coordinates (they can go outside the image with negative numbers)
    new_x1, new_y1 = cx - side // 2, cy - side // 2
    new_x2, new_y2 = cx + side // 2, cy + side // 2
    
    # We calculate how many pixels are "missing" on each side if we go outside the border
    pad_top, pad_bottom = max(0, -new_y1), max(0, new_y2 - h)
    pad_left, pad_right = max(0, -new_x1), max(0, new_x2 - w)
    
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

def get_retinaface_resize(frame, target_size=320):
    h, w = frame.shape[:2]
    side = max(h, w)
    
    padded = cv2.copyMakeBorder(
        frame, 
        0, side - h, 0, side - w, # Rellenar abajo y a la derecha
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # 2. Resize
    resized = cv2.resize(padded, (target_size, target_size))
    
    # El factor de escala es simplemente el tamaño original dividido por el nuevo
    scale = side / target_size
    
    return resized, scale

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
    predicted_ages = []
    while cap.isOpened():           # loop for processing each freame till the end of the video
        ret, frame = cap.read()     # frame = NumPy matrix, ret= boolean that indicates if the frame was read successfully or if we reached the end of the video
        if not ret:
            break
        frame_count += 1

        frame_res, scale = get_retinaface_resize(frame, target_size=320)
        # Detector is invoqued  
        detections = detector.detect(frame_res) #returns a list where each detection is a list of 5 elements: [x_min, y_min, x_max, y_max, confidence_score]
        for det in detections:
            x_min, y_min, x_max, y_max, score = det
            x_min, y_min, x_max, y_max = int(x_min * scale), int(y_min * scale), int(x_max * scale), int(y_max * scale)
            face_crop = get_square_crop_with_padding(frame, x_min, y_min, x_max, y_max, margin=1.3)

            if face_crop.size == 0:
                continue
                
            # Age estimation is invoqued
            age = int(round(estimator.estimate(face_crop)))
            predicted_ages.append(age)
            
            # Drawing the results on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x_min, y_min - 10), 
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
    return predicted_ages, frame_count

if __name__ == "__main__":
    run_pipeline(input_video_path, output_video_path)