import pandas as pd
import numpy as np
import json
import os
import torch
from images_MiVOLO_basic_pipeline import process_images_folder, MODEL_CONFIG
from sklearn.metrics import confusion_matrix


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_CONFIG = {
    "morph": {
        "dataset_path": os.path.join(BASE_DIR, "images_datasets", "CD2", "Album2"),
        #"dataset_path": r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\prueba\morph",
        "gt_path": os.path.join(BASE_DIR, "images_datasets", "CD2", "MORPH_Album2_comp.csv"),
        "inference_path": os.path.join(BASE_DIR, "images_datasets", "MORPH_MiVOLO_RESULTS_onlyface_v2.json"), 
        "type": "csv",
        "key_column": "photo"
    },
    "utk": {
        "dataset_path": os.path.join(BASE_DIR, "images_datasets", "UTKFace"), 
        #"dataset_path": r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\prueba\utk",
        "gt_path": os.path.join(BASE_DIR, "images_datasets", "UTK_face_groundtruth.json"),      
        "inference_path": os.path.join(BASE_DIR, "images_datasets", "UTK_MiVOLO_RESULTS_onlyface_v2.json"),
        "type": "json",
        "id_extractor": lambda filename: filename # El nombre es el ID directo
    }
}

#BASELINE_INF_TIME = 
#BASELINE_MAE = 
#BASELINE_MODEL_SIZE = 

CURRENT_DS = "morph" 
cfg = DATASET_CONFIG[CURRENT_DS]

total_time, vram_peak, num_files = process_images_folder(cfg["dataset_path"], save_photo=False)

if cfg["type"] == 'json':
    with open(cfg["gt_path"], 'r') as f: gt = json.load(f)
else:
    df_gt = pd.read_csv(cfg["gt_path"])
    df_gt['photo_clean'] = df_gt['photo'].apply(os.path.basename)
    
    # Droppin duplicated indexes
    df_gt = df_gt.drop_duplicates(subset=['photo_clean'])
    
    gt = df_gt.set_index('photo_clean')[['age', 'gender']].to_dict(orient='index')

with open(cfg["inference_path"], 'r') as f:
    results = json.load(f)

data = []
mapeo_genero = {'M': 'male', 'F': 'female'}
for filename, val in results.items():
    
    if filename in gt:
        real = gt[filename]
        pred = val

        p_gender = pred['gender'].lower()
        g_gender = mapeo_genero.get(real['gender'], real['gender']).lower()
        data.append({
            "filename": filename,
            "pred": pred['age'],
            "gt": real['age'],
            "abs_error": abs(pred['age'] - real['age']),
            "pred_gender": pred['gender'].lower(),
            "gt_gender": mapeo_genero.get(real['gender'], real['gender']).lower(),
            "is_correct": 1 if p_gender == g_gender else 0
        })
        
print(f"Total number of inference results: {len(results)}")
print(f"Total number of groundtruth-estimation founded: {len(data)}")

if len(data) == 0:
    print("Error! No match was found between the inferred files and the CSV.")
    print("Example of inferred filename:", list(results.keys())[0])
    print("Examples of 'photo' in CSV:", df_gt[cfg["key_column"]].head().tolist())
    exit()
else:
    df = pd.DataFrame(data)

# METRICS
mae = df["abs_error"].mean()
df["sq_error"] = (df["pred"] - df["gt"]) ** 2
rmse = np.sqrt(df["sq_error"].mean())

cs5 = (df["abs_error"] <= 5).mean() * 100
cs10 = (df["abs_error"] <= 10).mean() * 100


fps = num_files / total_time if total_time > 0 else 0
total_size_mb = 0
for key in MODEL_CONFIG:
    path = MODEL_CONFIG[key]["path"]
    if os.path.exists(path):
        total_size_mb += os.path.getsize(path) / (1024**2)

accuracy_per_mb = mae / total_size_mb
fps_per_mb = fps / total_size_mb
efficiency = fps / mae

#speed_up = BASELINE_INF_TIME / total_time
#mae_drop = BASELINE_MAE / mae
#comp_ratio = BASELINE_MODEL_SIZE / total_size_mb

summary_data = {
    "MAE": [mae],
    "RMSE" : [rmse],
    "FPS": [fps],
    "Total Inference Time": [total_time],
    "VRAM (MB)": [vram_peak],
    "Model Size (MB)": [total_size_mb],
    "CS5%" : [cs5],
    "CS10%" : [cs10],
    "MAE/Model_size(MB)" : [accuracy_per_mb],
    "FPS//Model_size(MB)" : [fps_per_mb],
    "Efficiency (FPS/MAE)" : [efficiency]
    #"Speed up" : [speed_up],
    #"MAE drop" : [mae_drop],
    #"Compression ratio" : [comp_ratio]
}

df_summary = pd.DataFrame(summary_data)

print("\n--- FINAL RESULTS ---")
print(df_summary)

gender_accuracy = df["is_correct"].mean() * 100

print(f"\n--- MÉTRICAS DE GÉNERO ---")
print(f"Gender Accuracy: {gender_accuracy:.2f}%")
cm = confusion_matrix(df["gt_gender"], df["pred_gender"], labels=["male", "female"])
print("\nConfusion matriz (Rows: Actual, Columns: Predicted):")
print(cm)