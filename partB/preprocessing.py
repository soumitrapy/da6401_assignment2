import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset,DataLoader

from torchvision import transforms # nice: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
#from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, path, class_names = None, transform=None, target_transform=None):
        super().__init__()
        self.path = path
        self.class_names = class_names
        self.transform = transform
        self.target_transform = target_transform

        if self.class_names is None:
            self.class_names =[x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
        
        self.images = []
        self.labels = []

        for i, cls in enumerate(self.class_names):
            img_dir = os.path.join(self.path, cls)
            for f in os.listdir(img_dir):
                if f.endswith(('.jpg','.png')):
                    self.images.append(os.path.join(img_dir, f))
                    self.labels.append(i)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




if __name__=="__main__":
    path = "inaturalist_12K/train"
    
    transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(), # not necessary
    ])
    ds = CustomDataset(path=path, transform=transform)
    dl = DataLoader(ds, batch_size=64)
    for x,y in dl:
        print(x.shape, y.shape)
        break
