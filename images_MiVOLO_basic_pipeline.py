import cv2
import torch
import os
from models_archive.estimator_mivolo_faceonly import MiVOLOAgeEstimator
import json
import time

# Configuration
MODEL_CONFIG = {
    "age": {
        # "path": "weights/model_imdb_cross_person_4.22_99.46.pth.tar", #for official test with face+body (results prcessed_data/resultado_100_ofoocial.jpg)
        #"path": "weights/model_only_age_imdb_4.29.pth.tar",
        "path" : "weights/model_utk_age_gender_4.23_97.69.pth.tar",
        #"path" : "weights/model_age_utk_4.23.pth.tar",
        "class": MiVOLOAgeEstimator
    }
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTH_JSON_PATH = os.path.join(BASE_DIR, "images_datasets","MORPH_MiVOLO_RESULTS_onlyface.json")
#OUTPUTH_JSON_PATH = os.path.join(BASE_DIR, "images_datasets","UTK_MiVOLO_RESULTS_onlyface.json")

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

# AT THE MOMENT IS FACE ONLY
def process_images_folder(folder_path, save_photo=False):
    output_folder = os.path.join(BASE_DIR, "images_datasets", "processed_data")    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # List with all the images within the folder
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    total_files = len(files)
    
    print(f"[*] {len(files)} images where found in '{folder_path}'")
    
    results_dict = {}
    total_inference_time = 0.0
    for i, file_name in enumerate(files, start=1):
        img_path = os.path.join(folder_path, file_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"[!] {file_name} Couldnt be load. Skipping...")
            continue

        h, w = frame.shape[:2]
        face_box = [0, 0, w, h]
        
        # Inference
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        predicted_age, predicted_gender, gender_score = estimator.estimate(frame, face_box)
        end_time = time.perf_counter()
        vram_peak = torch.cuda.max_memory_allocated() / (1024**2) # MB
        total_inference_time += (end_time - start_time)

        results_dict[file_name] = { #JSON
            "age": float(predicted_age),
            "gender": predicted_gender,
            "gender_score": gender_score 
        }
        
        if save_photo==True:
            cv2.putText(frame, f"Age: {predicted_age}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            output_path = os.path.join(output_folder, f"result_{file_name}")
            cv2.imwrite(output_path, frame)

        if i % 500 == 0 or i == total_files:
                porcentaje = (i / total_files) * 100
                print(f"[PROGRESS] Processed: {i}/{total_files} ({porcentaje:.1f}%) | Missing: {total_files - i}")

    if save_photo == True:
        print(f"[+] Process completed. Results saved in: '{output_folder}'")
    else:
        print(f"[+] Process completed.")

    with open(OUTPUTH_JSON_PATH, "w") as json_file:
        json.dump(results_dict, json_file, indent=2)
    print(f"[+] Results saved in: '{OUTPUTH_JSON_PATH}'")
    return total_inference_time, vram_peak, len(files)

if __name__ == "__main__":
    process_images_folder(r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\UTKFace")