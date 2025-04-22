import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(config):
    if config['name']=='resnet50':
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
        inp_size = m.fc.in_features
        m.fc = nn.Linear(inp_size, config['num_classes'])
        return m
    else:
        return NotImplemented


