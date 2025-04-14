import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path):
    model = models.efficientnet_b0(weights=None)  # Use weights=None to avoid downloading
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5)  # 5 severity classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

