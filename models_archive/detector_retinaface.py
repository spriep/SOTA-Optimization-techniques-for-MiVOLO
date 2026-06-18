# DEPRACATED USE THE RETINAFACE_ONNX INSTEAD !!!!!!
import os
import sys
import torch
import numpy as np
import cv2

# path to find the RetinaFace folder and inject it into Python without failing to find the next imports
# os.path.abspath and os.path.join --> converts it into a clean path depending on your OS
# os.path.dirname(__file__) -->  in C:/.../basic_pipeline/models_archive/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pytorch_Retinaface')))

from models.retinaface import RetinaFace
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms
from models_archive.base_model import BaseDetector

class RetinaFaceDetector(BaseDetector):
    def __init__(self, weights_path, device):
        self.device = device        # CUDA or CPU
        self.cfg = cfg_mnet         #dictionary with the architectures parameters of the MobileNet version of RetinaFace (/RetinaFace/data/config.py)
        
        
        current_dir = os.getcwd()                                                                       # Saving the root path: basic_pipeline
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pytorch_Retinaface'))
        os.chdir(repo_dir)                                                                              # we enter temporaly to the repository to avoid path errors with the RetinaFace folder
        self.net = RetinaFace(cfg=self.cfg, phase='test')                                               # creating the neural network, its internal contructor searches for the weights folder
        os.chdir(current_dir)                                                                           # we return to the root path to continue running the rest of the script
        
        # Load official weights 
        state_dict = torch.load(weights_path, map_location=device) # with the architecture layers and its weights
        from collections import OrderedDict
        new_state_dict = OrderedDict()                             # Ordered dictionary to store the cleaned weights 
        for k, v in state_dict.items():               # Clean up the weight dictionary in case it was trained in parallel (im using a RTX 3050)
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        self.net.to(device)                           # Upload the entire model to the GPU cores
        self.net.eval()                               # Turn off training layers (such as Dropout or BatchNorm)
        self.priors_precalculated = False             # Flag indicating whether priors have been precalculated for the current video frame size (for the first frame is false)

    def detect(self, frame):
        frame_h, frame_w, _ = frame.shape
        
        # Precalculate (PriorBoxes) for the native video size "cuadridula invisible"/"invisible grid"
        # This avoids recalculating them on each frame and saves a lot of CPU/GPU
        if not self.priors_precalculated: #Only calculated on the first frame
            priorbox = PriorBox(self.cfg, image_size=(frame_h, frame_w))
            with torch.no_grad():
                self.priors = priorbox.forward().to(self.device) #uploaded to the self.device
            self.priors_precalculated = True

        # Preprocesing
        img = np.float32(frame)                                     # OpenCv reads the image in uint8 format, we convert it to float32 to make the normalization
        img -= (104, 117, 123)                                      # Normalization (mean 0 with ImageNet findings) to stabilize the training and inference 
        img = img.transpose(2, 0, 1)                                # adaptation to neurons input format
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)    #loading the pytorch tensors (converted matrix) to the GPU and adding a batch dimension (1 frame)

        with torch.no_grad():                                                           # disabling gradient calculation for inference (saves memory and speeds up computations)
            loc, conf, _ = self.net(img)                                                # forward pass through the network, obtaining the location and confidence predictions            
            scale = torch.Tensor([frame_w, frame_h, frame_w, frame_h]).to(self.device)  
            boxes = decode(loc.data.squeeze(0), self.priors.data, self.cfg['variance']) # aplying loc and conf to self.priors to get the candidate bounding boxes (frame_w, frame_h)
            boxes = boxes * scale                                                       #transformation to real coordenates on screen
            boxes = boxes.cpu().numpy()
            scores = conf.data.squeeze(0).cpu().numpy()[:, 1]

            inds = np.where(scores > 0.6)[0]    # Rectangles with less than 60% confidence are discarded
            boxes = boxes[inds]
            scores = scores[inds]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.4)        # Non-Maximum Suppression (NMS) to remove overlapping boxes, keeping only the one with the highest confidence score. The threshold is set to 40% of overlaping area.
            dets = dets[keep, :]
        return dets # returns [[x1, y1, x2, y2, score], ...]