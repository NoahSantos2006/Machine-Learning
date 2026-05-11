import torch
import torchvision
from torchvision.io import read_image
from torch import nn
from torchvision import transforms

import matplotlib.pyplot as plt
import json
import os
import re

def load_data(text_path, image_path):

    for root, dirs, files in os.walk(text_path):
        
        continue
    
    regex = "[a-zA-Z]+"
    labels = []
    
    for file in files:

        with open(os.path.join(root, file), "r") as f:

            curr_text = f.read()
            find = re.findall(regex, curr_text)
            labels.append(" ".join(find))

    for root, dirs, files in os.walk(image_path):
        
        continue
    
    images = []
    # transform the image to keep a consistent shape
    transform = transforms.Compose([
        transforms.Resize((192, 256))
    ])
    
    for file in files:

        img = torchvision.io.read_image(os.path.join(root, file))
        img = transform(img)
        images.append(img)
    
    training_data = list(zip(images, labels))

    return training_data

