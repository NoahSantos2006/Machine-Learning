import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

def load_data():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_data = datasets.EMNIST(
        root="data",
        split="letters",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.EMNIST(
        root="data",
        split="letters",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32)

    return train_dataloader, test_dataloader, len(train_data.classes)

if __name__ == "__main__":

    train_data, test_data = load_data()