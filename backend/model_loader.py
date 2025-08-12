import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ContrastiveMobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.encoder = models.mobilenet_v2(weights='DEFAULT').features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_projection = nn.Sequential(
            nn.Linear(1280, 2048),
            nn.ReLU(inplace=True)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128)
        )
        
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_embedding=False, return_projection=False):
        features = self.encoder(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        
        r = self.feature_projection(features)
        r = F.normalize(r, dim=1)
        
        if return_embedding:
            return r
        elif return_projection:
            z = self.projection(r)
            z = F.normalize(z, dim=1)
            return z
        else:
            return self.classifier(r)

def load_model(model_path, device='cpu'):
    model = ContrastiveMobileNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_lesion_info():
    """Return lesion type information"""
    return {
        0: {
            'name': 'Melanoma (MEL)', 
            'description': 'Malignant skin tumor - requires immediate medical attention',
            'severity': 'high',
            'color': '#dc3545'
        },
        1: {
            'name': 'Nevus (NV)', 
            'description': 'Benign mole - generally harmless but monitor for changes',
            'severity': 'low',
            'color': '#28a745'
        },
        2: {
            'name': 'Basal Cell Carcinoma (BCC)', 
            'description': 'Most common skin cancer - treatable when caught early',
            'severity': 'medium',
            'color': '#fd7e14'
        },
        3: {
            'name': 'Actinic Keratosis (AKIEC)', 
            'description': 'Pre-cancerous lesion - should be monitored by dermatologist',
            'severity': 'medium',
            'color': '#ffc107'
        },
        4: {
            'name': 'Benign Keratosis (BKL)', 
            'description': 'Benign skin growth - usually no treatment needed',
            'severity': 'low',
            'color': '#28a745'
        },
        5: {
            'name': 'Dermatofibroma (DF)', 
            'description': 'Benign skin nodule - typically harmless',
            'severity': 'low',
            'color': '#28a745'
        },
        6: {
            'name': 'Vascular Lesion (VASC)', 
            'description': 'Blood vessel lesion - usually benign',
            'severity': 'low',
            'color': '#17a2b8'
        }
    }