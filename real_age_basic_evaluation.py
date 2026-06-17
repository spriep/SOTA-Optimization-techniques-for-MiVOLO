import os
import json
import time
import pandas as pd
import torch
from basic_pipeline import run_pipeline

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, "Real_Age_Faces_Dataset", "real_age_2026_v2.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "Real_Age_Faces_Dataset", "processed_data", "diezsei_junio")

def get_ground_truth(video_filename, metadata):
    video_id = os.path.splitext(video_filename)[0]
    for entry in metadata:
        if entry['video_id'] == video_id:
            return entry
    return None

def run_benchmark(video_paths):
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    results = []
    
    for path in video_paths:
        filename = os.path.basename(path)
        gt = get_ground_truth(filename, metadata)
        
        if not gt:
            print(f"[!] No se encontró metadata para: {filename}")
            continue

        print(f"[*] Evaluando: {filename} (Edad Real: {gt['age']})")
        
        # Medir tiempo de ejecución
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Llamada al pipeline modular refactorizado
        output_path = os.path.join(OUTPUT_DIR, f"null_{filename}")
        detected_ages, total_frames = run_pipeline(path, output_path, save_video=False)
        
        duration = time.time() - start_time
        fps = total_frames / duration if duration > 0 else 0
        vram_peak = torch.cuda.max_memory_allocated() / (1024**2) # MB
        
        # Cálculo de métricas
        mae = 0
        mean_pred_age = 0
        if detected_ages:
            mae = sum([abs(a - gt['age']) for a in detected_ages]) / len(detected_ages)
            mean_pred_age = sum(detected_ages) / len(detected_ages)
        
        results.append({
            "video": filename,
            "fps": fps,
            "vram_mb": vram_peak,
            "mae": mae,
            "mean_predicted_age": round(mean_pred_age, 2),
        })
        
    print("\n--- VIDEO DETAILS ---")
    print(pd.DataFrame(results))
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Asegurar que el directorio de salida exista
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
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
    
    print("[*] Iniciando Benchmark Modular...")
    df_results = run_benchmark(test_videos)
    
    # Showing output
    df_summary = df_results.mean(numeric_only=True).drop(labels=["mean_predicted_age"])

    print("\n--- FINAL RESULTS ---")
    print(df_summary)
    
    '''# Exportar a Excel
    output_report = "modular_benchmark_report.xlsx"
    df_results.to_excel(output_report, index=False)
    print(f"[+] Resultados guardados en {output_report}")'''