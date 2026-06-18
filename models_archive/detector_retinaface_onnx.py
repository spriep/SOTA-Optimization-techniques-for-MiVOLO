import os
import sys
import torch
import numpy as np
import cv2
import onnxruntime as ort # IMPORTANTE: Importing ONNX engine


# path to find the RetinaFace folder and inject it into Python without failing to find the next imports
# os.path.abspath and os.path.join --> converts it into a clean path depending on your OS
# os.path.dirname(__file__) -->  in C:/.../basic_pipeline/models_archive/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pytorch_Retinaface')))
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms
from models_archive.base_model import BaseDetector

class RetinaFaceDetector(BaseDetector):
    def __init__(self, onnx_path, device):
        self.device = device    # CUDA or CPU
        self.cfg = cfg_mnet     #dictionary with the architectures parameters of the MobileNet version of RetinaFace (/RetinaFace/data/config.py)
        self.priors_precalculated = False #the ancores (rectangles within the image grid)
        
        # STARTING ONNX SESION  (optimized inference motor)
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        # We obtain the model name so we can send it the inputs. 
        self.input_name = self.session.get_inputs()[0].name
        
        target_size = self.cfg['image_size']
        priorbox = PriorBox(self.cfg, image_size=(target_size, target_size))
        with torch.no_grad():
            self.priors = priorbox.forward().to(self.device)
        print(f"[+] Motor ONNX cargado correctamente usando: {providers[0]}")
    
    def _preprocess(self, frame):
        # Resizing with config.py poaramemeters to the size expected by the backbone (640x640)
        target_size = self.cfg['image_size']
        img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, frame):
        frame_h, frame_w, _ = frame.shape

        # Preprocessing (native resize + normalization)
        img_input = self._preprocess(frame)
        
        # Inference
        loc, conf, _ = self.session.run(None, {self.input_name: img_input})
        
       # Post-procesing
        loc_tensor = torch.from_numpy(loc).to(self.device)
        conf_tensor = torch.from_numpy(conf).to(self.device)

        # We use the pre-calculated priors in __init__ to decode
        # We scaled to the original size (frame_w, frame_h)
        scale = torch.Tensor([frame_w, frame_h, frame_w, frame_h]).to(self.device)
        
        boxes = decode(loc_tensor.squeeze(0), self.priors.data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf_tensor.squeeze(0).cpu().numpy()[:, 1]

        # Filtered by confidence threshold (in. 0.6)
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # Output format: [x1, y1, x2, y2, score]ore]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        
        # NMS (Non-Maximum Suppression)
        keep = py_cpu_nms(dets, 0.4)
        
        return dets[keep, :]  # returns [[x1, y1, x2, y2, score], ...]