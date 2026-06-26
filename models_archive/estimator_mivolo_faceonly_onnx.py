import onnxruntime as ort
import numpy as np
import cv2
import os

class MiVOLOOnnxEstimator:
    def __init__(self, weights_path, device, quant_type=None):
        self.device_name = 'cuda' if 'cuda' in str(device) else 'cpu'
        
        # 1. Configuración de ejecución (Prioriza CUDA si está disponible)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device_name == 'cuda' else ['CPUExecutionProvider']
        
        print(f"[*] Inicializando ONNX Runtime en: {self.device_name.upper()}...")
        self.session = ort.InferenceSession(weights_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        
        # Variables de normalización (estándar de MiVOLO)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Variables de edad (necesitarás extraerlas manualmente del archivo .pth 
        # original una vez y hardcodearlas aquí o pasarlas como json)
        self.min_age, self.max_age, self.avg_age = 0.0, 100.0, 40.0 
        
    def estimate(self, frame, face_box):
        # 1. Recorte y preprocesamiento idéntico al original
        x1, y1, x2, y2 = [int(v) for v in face_box]
        face_crop = frame[max(0, y1):y2, max(0, x1):x2]
        
        if face_crop.size == 0: return 0.0, "unknown", 0.0

        # 2. Preprocesamiento (resize, norm, BGR to RGB, CHW)
        img = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1)) # (3, 224, 224)
        
        # 2. Expandir a 6 canales (concatenación necesaria para este modelo ONNX)
        img = np.concatenate([img, img], axis=0) # (6, 224, 224)
        
        # 3. Añadir dimensión de Batch (1, 6, 224, 224)
        img = np.expand_dims(img, axis=0)
        
        # 4. Inferencia
        output = self.session.run(None, {self.input_name: img})[0]
        
        # 4. Post-procesamiento
        raw_age = output[0, 2] if output.shape[1] > 1 else output[0, 0]
        predicted_age = raw_age * (self.max_age - self.min_age) + self.avg_age
        
        gender_probs = np.exp(output[0, :2]) / np.sum(np.exp(output[0, :2]))
        predicted_gender = "male" if gender_probs[0] > gender_probs[1] else "female"
        gender_score = float(max(gender_probs))
        
        return round(max(0.0, predicted_age), 1), predicted_gender, round(gender_score, 1)