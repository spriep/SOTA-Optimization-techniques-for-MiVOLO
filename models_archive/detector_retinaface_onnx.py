import os
import sys
import torch
import numpy as np
import cv2
import onnxruntime as ort # IMPORTANTE: Importamos el motor ONNX

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
        self.priors_precalculated = False
        
        # STARTING ONNX SESION  (optimized inference motor)
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # We obtain the model name so we can send it the inputs. 
        self.input_name = self.session.get_inputs()[0].name
        print(f"[+] Motor ONNX cargado correctamente usando: {providers[0]}")

    def detect(self, frame):
        frame_h, frame_w, _ = frame.shape
        
        # Precalculate (PriorBoxes) for the native video size "cuadridula invisible"/"invisible grid"
        # This avoids recalculating them on each frame and saves a lot of CPU/GPU
        if not self.priors_precalculated:   #Only calculated on the first frame
            priorbox = PriorBox(self.cfg, image_size=(frame_h, frame_w))
            with torch.no_grad():
                self.priors = priorbox.forward().to(self.device)
            self.priors_precalculated = True

        
        # PREPROCESING FOR ONNX (everything NumPy, not Torch)
       
        img = np.float32(frame)            # OpenCv reads the image in uint8 format, we convert it to float32 to make the normalization
        img -= (104, 117, 123)             # Normalization (mean 0 with ImageNet findings) to stabilize the training and inference
        img = img.transpose(2, 0, 1)       # adaptation to neurons input format
        img = np.expand_dims(img, axis=0)  # Adding Batch Size of 1, so onnx knows it will be trating groups of 1 image
        
        # 2. INFERENCE WITH ONNX RUNTIME (faster than PyTorch)
        # obtaining the outputs of the model (loc, conf and landmarks)
        loc, conf, _ = self.session.run(None, {self.input_name: img})

        # 3. POST-PROCESING (Using Torch for decoding)
        # ONNX returns NumPy arrays. Since the original author's 'decode' function 
        # is programmed to use PyTorch tensors, we convert them back.
        loc_tensor = torch.from_numpy(loc).to(self.device)
        conf_tensor = torch.from_numpy(conf).to(self.device)

        scale = torch.Tensor([frame_w, frame_h, frame_w, frame_h]).to(self.device)
        
        # aplying loc and conf to self.priors to get the candidate bounding boxes (frame_w, frame_h)
        # and transformation to real coordenates on screen
        boxes = decode(loc_tensor.squeeze(0), self.priors.data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf_tensor.squeeze(0).cpu().numpy()[:, 1]

        # confidence thresholding
        # Rectangles with less than 60% confidence are discarded
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # Non-Maximum Suppression (NMS) to remove overlapping boxes, keeping only the one with the highest confidence score. The threshold is set to 40% of overlaping area.
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
            
        return dets # returns [[x1, y1, x2, y2, score], ...]