import pandas as pd
import numpy as np
import json
import os
import torch
from images_MiVOLO_basic_pipeline import process_images_folder, MODEL_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "images_datasets", "CD2", "Album2")
#DATASET = os.path.join(BASE_DIR, "images_datasets", "UTKFace")
#DATASET = r"C:\Users\saioa\Desktop\resi_GVIS\basic_pipeline\images_datasets\prueba\morph"
GROUNDTRUTH = os.path.join(BASE_DIR, "images_datasets", "UTK_face_groundtruth.json")
INFERENCE_RESULTS = os.path.join(BASE_DIR, "images_datasets", "UTK_MiVOLO_RESULTS_onlyface_basic.json")

#BASELINE_INF_TIME = 
#BASELINE_MAE = 
#BASELINE_MODEL_SIZE = 

total_time, vram_peak, num_files = process_images_folder(DATASET, save_photo=False)

ext = os.path.splitext(GROUNDTRUTH)[1].lower()
if ext == '.json':
    with open(GROUNDTRUTH, 'r') as f:
        gt = json.load(f)
elif ext == '.csv':
    df = pd.read_csv(GROUNDTRUTH)
    gt = df.set_index('id_num')['age'].to_dict()

with open(INFERENCE_RESULTS, 'r') as f:
        results = json.load(f)


data = []
for filename, val in results.items():
    if filename in gt:
        pred = val['age']
        real = gt[filename]['age']
        data.append({
            "filename": filename,
            "pred": pred,
            "gt": real,
            "abs_error": abs(pred - real)
        })
        
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