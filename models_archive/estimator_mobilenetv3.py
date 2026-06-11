import torch
import torch.nn.functional as F

import cv2
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from models_archive.base_model import BaseEstimator

class MobileNetAgeEstimator(BaseEstimator):
    def __init__(self, weights_path, device):
        self.device = device
        
        # 1. Load MobileNetV3 Large with its official object detection weights
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=weights)
        
        # 2. Age-specific classifier surgery
        # Unlike ResNet (which has a single .fc layer), MobileNetV3 has a sequential block in .classifier: [Linear, Hardswish, Dropout, Linear]
        # We modify the last linear layer 
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = torch.nn.Linear(in_features, 116) # 116 because we are using the ordinal regression approach with 116 age classes (0-115)
        
        # 3. Age-specific weight loading (for when you have your own .pth)
        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            
            # Loading weights
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            # Cleaning the downloaded weights for MobileNetV3 (they usually come with a "net." prefix that causes errors when loading)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # deletimng "net." prefix
                name = k[4:] if k.startswith('net.') else k
                # deleting "module." prefix 
                if name.startswith('module.'):
                    name = name[7:]
                new_state_dict[name] = v
                
            # We loaded the clean weights. using strict=False to ignore training metadata
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[+] Age weights loaded into MobileNetV3 from: {weights_path}")
        else:
            print("[!] Warning: Using untrained final layer of MobileNetV3 (will predict around 0).")
            
        self.model.to(device)
        self.model.eval()

        # 4. Standard ImageNet transformations (compatible with MobileNetV3)
        '''self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])'''

        # Tranformation to match the dowloaadeed weight training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Using a neutral centered transformation intead of the imagenet one
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # We precompute the vector of ages [0, 1, 2, ..., 115] and upload it to the GPU
        self.age_classes = torch.arange(0, 116).to(device).float()

    def estimate(self, face_crop):
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor_face = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. We scanned the model and it returned 116 raw scores
            raw_scores = self.model(tensor_face)
            
            # 2. We convert those raw scores into probabilities using sigmoid [0.0, 1.0]
            probabilities = torch.sigmoid(raw_scores)
            
            # 3. if the probability of being greater than 0.5 is high, we consider that the person is older than that age class. We sum all the age classes that are predicted as "older" to get the final age estimation.
            predicted_age = torch.sum(probabilities > 0.5, dim=1)
            
            return predicted_age.item()