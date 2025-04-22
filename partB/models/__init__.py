import torch
import torch.nn as nn
import torch.nn.functional as F
from .smallcnn import SmallCNN

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class DefaultModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv = nn.Sequential(
            nn.Conv2d(self.config['in_channels'], self.config['filters'], self.config['kernel_size']),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.config['filters']*127*127, self.config['num_classes'])#self.config['dense_neurons'])
        #self.fc2 = nn.Linear(self.config['dense_neurons'], self.config['num_classes'])
    def forward(self,x):
        x = self.conv(x)
        x = self.flatten(x)  # (B, C)
        x = self.activation(x)
        x = self.fc1(x)
        # x = self.activation(x)
        # x = self.fc2(x)
        return x


def get_model(config):
    if config['name']=='resnet50':
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
        inp_size = m.fc.in_features
        m.fc = nn.Linear(inp_size, config['num_classes'])
        return m
    else:
        return NotImplemented