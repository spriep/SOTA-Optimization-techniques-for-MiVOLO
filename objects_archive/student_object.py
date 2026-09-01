import torch
import torch.nn as nn
import timm

class StaticStudentMiVOLO(nn.Module):
    def __init__(self, num_outputs=3):
        super().__init__()
        #self.backbone = timm.create_model('mobilenetv3_small_050', pretrained=False, num_classes=0)
        #2self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        #self.backbone = timm.create_model('efficientformer_l1', pretrained=True, num_classes=0)
        #1self.backbone = timm.create_model('mobilevit_xs', pretrained=True, num_classes=0)
        #self.backbone = timm.create_model('spnasnet_100', pretrained=True, num_classes=0)
        self.backbone = timm.create_model('ghostnet_100', pretrained=True, num_classes=0)
        #self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        # CAMBIO: Usamos una red pequeña para la cabeza de edad para mayor estabilidad
        self.age_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            #nn.Tanh(), # Forzamos el rango a [-1, 1] para que coincida con la escala del logit
        )
        
        self.gender_head = nn.Linear(num_features, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features), self.gender_head(features)