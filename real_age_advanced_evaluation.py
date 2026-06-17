import json
import os
import torch
import pandas as pd
import numpy as np
import time

from advanced_pipeline import run_pipeline

from advanced_pipeline import MODEL_CONFIG

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "Real_Age_Faces_Dataset", "real_age_2026_v2.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "Real_Age_Faces_Dataset", "processed_data", "diezsei_junio")

# Loading the metada (.json)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

def get_ground_truth(video_filename):
    # Search your JSON file for video information based on the file name
    video_id = os.path.splitext(video_filename)[0]
    for entry in metadata:
        if entry['video_id'] == video_id:
            return entry
    return None

def run_benchmark(video_paths):
    results = []
    
    for path in video_paths:
        filename = os.path.basename(path)
        gt = get_ground_truth(filename)
        
        if not gt:
            print(f"No metadata was found for {filename}")
            continue

        print(f"[*] Evaluating: {filename} (Ral Age: {gt['age']})")

        # Measuring VRAM and FPS
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Executing pipeline
        detected_ages, total_frames = run_pipeline(path, os.path.join(OUTPUT_DIR, f"null_{filename}"), save_video=False)
        
        duration = time.time() - start_time
        fps = total_frames / duration if duration > 0 else 0
        vram_peak = torch.cuda.max_memory_allocated() / (1024**2) # MB

        # 2. MAE for Age
        #print(detected_ages)
        mae = np.nan
        mean_age = np.nan

        if len(detected_ages) > 0:
            mae = np.mean([abs(a - gt['age']) for a in detected_ages])
            mean_age = np.mean(detected_ages)
        else:
            print(f"[!] Aviso: No se detectaron caras en {filename}. Saltando métricas.")
            mae = 0

        # 3. Model Size 
        total_size_mb = 0
        for key in MODEL_CONFIG:
            path = MODEL_CONFIG[key]["path"]
            if os.path.exists(path):
                total_size_mb += os.path.getsize(path) / (1024**2)

        # 4. mAP (which groudtruth)
        # mAP = calcular_map(detected_boxes, gt_boxes) 

        results.append({
            "video": filename,
            "fps": fps,
            "mae": mae,
            "vram_mb": vram_peak,
            "model_size_mb": total_size_mb,
            "mean predicted age": mean_age
            #"mAP": mAP
        })

    print("\n--- VIDEO DETAILS ---")
    print(pd.DataFrame(results))
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    test_videos = [
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/22yr_m_c_px_007.mp4", 
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/21yr_f_l_kl_022.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/20yr_f_c_ke_098.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/19yr_f_c_kd_080.mp4" ,
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/18yr_m_i_sy_008.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/17yr_f_a_xv_016.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/16yr_m_c_nc_105.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/15yr_f_i_hs_007.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/14yr_m_a_aex_008.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/13yr_f_b_fg_003.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/12yr_m_b_av_006_bq.mp4",
        "Real_Age_Faces_Dataset/01_25_Recopilacion_videos_redes/11yr_f_l_iv_006.mp4"
    ]
    
    print("[*] Initiating Benchmark...")
    df_results = run_benchmark(test_videos)
    
    # Showing output
    df_summary = df_results.mean(numeric_only=True).drop(labels=["mean predicted age"])

    print("\n--- FINAL RESULTS ---")
    print(df_summary)
    
    # exporting
    '''df_results.to_csv("final_benchmark_report.csv", index=False)
    print("[+] Reporte guardado en final_benchmark_report.csv")'''

    '''output_file = "evaluation_metrics_reports/final_benchmark_report.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Detalle_Videos', index=False)
        df_summary.to_excel(writer, sheet_name='Resumen_Final', index=False)
    
    print(f"[+] Resultados exportados profesionalmente a {output_file}")'''


    