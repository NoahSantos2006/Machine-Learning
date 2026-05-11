import torch
import torchvision
from torch import nn

import matplotlib.pyplot as plt


if __name__ == "__main__":

    img = torchvision.io.read_image("brandon.jpg")
    
    print(img.shape)
    exit()

    fig = plt.figure(figsize=(10, 7))
    plt.imshow(img)

    img = img.unsqueeze(dim=0).type(torch.float32)

    cnn_layer = nn.Conv2d(in_channels=1792, out_channels=10, kernel_size=3)

    img = cnn_layer(img)

    print(img)