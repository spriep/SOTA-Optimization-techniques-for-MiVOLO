import torch
import cv2
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from models_archive.base_model import BaseEstimator

class ResNetAgeEstimator(BaseEstimator):
    def __init__(self, weights_path, device):
        self.device = device # CUDA or CPU
        # Note: I'm using the default weights here, but if you have your own .pth file with your age experiments in the future,
        # you would change weights=None and include your torch.load
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)  #we insert to our model the ResNet50 downloaded weights
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1) # we change the output layer to match our problems solution (1 single age)
        self.model.to(device)
        self.model.eval() #Activate evaluation mode (turn off training layers like Dropout)

        # Adaptation pipeline for the face crops before feeding them to the ResNet50
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def estimate(self, face_crop):
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)                  # sincronyzing formats
        tensor_face = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)    # adapting the face crop and adding a batch dimension (1 face crop)
        
        with torch.no_grad():
            age_prediction = self.model(tensor_face)
            return age_prediction.item()