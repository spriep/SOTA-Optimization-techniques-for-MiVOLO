import os
import sys
import torch
import cv2
import torchvision.transforms as transforms
from models_archive.base_model import BaseEstimator
from torchvision.transforms import InterpolationMode
from mivolo.data.misc import prepare_classification_images
import onnxruntime as ort



# mivolo folder path configuration for importing the MiVOLO model 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importimg the MiVOLO model class from the mivolo folder
try:
    from mivolo.model.mi_volo import MiVOLO
except ImportError as e:
    print("[!] FATAL ERROR: 'mivolo' folder not found in the project root.")
    print(e)
    sys.exit(1)

class MiVOLOAgeEstimator(BaseEstimator):
    def __init__(self, weights_path, device, quant_type=None):
        self.device = device
        self.quant_type = quant_type


        print("[*] Extracting normalization variables from the file...")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.min_age = float(ckpt['min_age'])
        self.max_age = float(ckpt['max_age'])
        self.avg_age = float(ckpt['avg_age'])

        print(f"[*] Initialising MiVOLO (Vision Transformer) on {str(self.device).upper()}...")

        # Initialize the official wrapper for MiVOLO (Constructs the architecture + load the weights + prepares the onlyface mode)
        # The class constructs the transformer and load the weights
        self.mivolo_wrapper = MiVOLO(
            weights_path, 
            device=str(self.device),
            half = True if self.quant_type == 'dynamicPTQfp16' else False # True --> half precisión (FP16) ; False --> full precisión (FP32) ----------------------------------------------------
        ) 
        # Extracting the mathematical model for inference
        self.model = self.mivolo_wrapper.model
        self.model.eval()

        if self.quant_type == 'dynamicPTQfp16':
            print("[*] Applying FP16 (Half Precision)...")

        elif self.quant_type is not None:
            print(f"[!] Notice: The quantization mode '{self.quant_type}' is not supported.")
        
        self.model.to(self.device)
        try:
            print(f"[*] Precisión de pesos DESPUÉS: {self.model.network[0][0].attn.v.weight.dtype}")
        except:
            print("[*] Precisión de pesos DESPUÉS: Modelo cuantizado (INT8/Packed)")


        # Standard normalization of ImageNet for the Transformers 
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_square_crop_with_padding(self, frame, x_min, y_min, x_max, y_max, margin=1.3):
        h, w = frame.shape[:2]
        box_w, box_h = x_max - x_min, y_max - y_min
        cx, cy = x_min + box_w // 2, y_min + box_h // 2
        side = int(max(box_w, box_h) * margin)
        
        new_x1, new_y1 = cx - side // 2, cy - side // 2
        new_x2, new_y2 = cx + side // 2, cy + side // 2
        
        pad_top, pad_bottom = max(0, -new_y1), max(0, new_y2 - h)
        pad_left, pad_right = max(0, -new_x1), max(0, new_x2 - w)
        
        valid_x1, valid_y1 = max(0, new_x1), max(0, new_y1)
        valid_x2, valid_y2 = min(w, new_x2), min(h, new_y2)
        
        crop_valid = frame[valid_y1:valid_y2, valid_x1:valid_x2]
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            return cv2.copyMakeBorder(crop_valid, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return crop_valid
    
    def estimate(self, frame, face_box):
        def crop_img(f, box):
            # cropping the frame with the face_box
            x1, y1, x2, y2 = [int(v) for v in box]
            h, w = f.shape[:2]
            # making sure limits are within the frame
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            return f[y1:y2, x1:x2]

        face_crop = crop_img(frame, face_box)

        if face_crop.size == 0:
            return None

        # Use MiVOLO's native tools to prepare the image
        mean = self.mivolo_wrapper.data_config["mean"]
        std = self.mivolo_wrapper.data_config["std"]
        input_size = self.mivolo_wrapper.input_size 

        # prepare_classification_images performs resizing (bicubic), normalization, and ToTensor
        tensor_face = prepare_classification_images([face_crop], input_size, mean, std, device=self.device)
        if self.quant_type == 'dynamicPTQfp16':
            tensor_face = tensor_face.half()
            with torch.no_grad():
                output = self.model(tensor_face)
        else: 
            with torch.no_grad():
                # HERE MiVOLO COULD concatenate the face and body in dimension 1 (channels)
                output = self.model(tensor_face)
                
        # Post-procesing
        raw_age = output[0, 2].item() if output.shape[1] > 1 else output[0, 0].item()
        predicted_age = raw_age * (self.max_age - self.min_age) + self.avg_age

        gender_output = output[:, :2].softmax(-1)

        gender_probs, gender_indx = gender_output.topk(1)
        predicted_gender = "male" if gender_indx.item() == 0 else "female"
        gender_score = gender_probs.item()

        return round(max(0.0, predicted_age), 1), predicted_gender, round(float(gender_score),1)
