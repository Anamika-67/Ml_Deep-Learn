import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, model_type="resnet50"):
        super().__init__()
        self.model_type = model_type
        if model_type == "resnet50":
            base_model = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.fc = nn.Linear(2048, 512)
        elif model_type == "efficientnet":
            base_model = models.efficientnet_b0(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.fc = nn.Linear(1280, 512)
        elif model_type == "mobilenet":
            base_model = models.mobilenet_v2(pretrained=True)
            self.feature_extractor = base_model.features
            self.fc = nn.Linear(1280, 512)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        features = self.feature_extractor(x)
        
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            
        features = features.view(B*T, -1)
        features = self.fc(features)
        features = features.view(B, T, 512)
        return features
