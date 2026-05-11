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

    return train_dataloader, test_dataloader

if __name__ == "__main__":

    num_to_letter = {i: chr(64 + i) for i in range(1, 27)}

    train_data, test_data = load_data()

