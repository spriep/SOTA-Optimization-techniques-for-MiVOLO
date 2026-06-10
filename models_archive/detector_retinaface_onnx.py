import os
import sys
import torch
import numpy as np
import cv2
import onnxruntime as ort # IMPORTANTE: Importamos el motor ONNX

# El inyector de rutas se queda igual para importar cfg_mnet y la función NMS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pytorch_Retinaface')))
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms
from models_archive.base_model import BaseDetector

class RetinaFaceDetector(BaseDetector):
    def __init__(self, onnx_path, device):
        self.device = device
        self.cfg = cfg_mnet
        self.priors_precalculated = False
        
        # INICIAR LA SESIÓN ONNX (El motor de inferencia optimizado)
        # Selecciona la GPU si device es 'cuda', si no, usa la CPU
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Obtenemos el nombre exacto de la entrada que ONNX espera
        self.input_name = self.session.get_inputs()[0].name
        print(f"[+] Motor ONNX cargado correctamente usando: {providers[0]}")

    def detect(self, frame):
        frame_h, frame_w, _ = frame.shape
        
        # (Esto se queda igual) Precalcular la cuadrícula invisible
        if not self.priors_precalculated:
            priorbox = PriorBox(self.cfg, image_size=(frame_h, frame_w))
            with torch.no_grad():
                self.priors = priorbox.forward().to(self.device)
            self.priors_precalculated = True

        # =========================================================
        # 1. PREPROCESAMIENTO PARA ONNX (Todo en NumPy, nada de Torch)
        # =========================================================
        img = np.float32(frame)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0) # Añade el Batch Size de 1
        
        # =========================================================
        # 2. INFERENCIA CON ONNX RUNTIME (Ultra rápido)
        # =========================================================
        # Le pasamos un diccionario con el nombre de la entrada y la imagen
        loc, conf, _ = self.session.run(None, {self.input_name: img})

        # =========================================================
        # 3. POST-PROCESAMIENTO (Volvemos a Torch solo para decodificar)
        # =========================================================
        # ONNX nos devuelve NumPy arrays. Como la función 'decode' del autor original 
        # está programada para usar tensores de PyTorch, los convertimos de vuelta.
        loc_tensor = torch.from_numpy(loc).to(self.device)
        conf_tensor = torch.from_numpy(conf).to(self.device)

        scale = torch.Tensor([frame_w, frame_h, frame_w, frame_h]).to(self.device)
        
        # Decodificación geométrica
        boxes = decode(loc_tensor.squeeze(0), self.priors.data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf_tensor.squeeze(0).cpu().numpy()[:, 1]

        # Filtro de confianza
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
            
        return dets