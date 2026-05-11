import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import json
import os
import re

from loading_dataset import load_data

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

class TextRecognitionModel(nn.Module):

    def __init__(self, input_shape: int, hidden_features: int, output_shape: int):

        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_features, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_shape=hidden_features, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features, out_features=output_shape)
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

def train_step(model, dataset, epochs, accuracy_fn, loss_fn, optimizer):

    for epoch in range(epochs):

        for batch, (X, y) in enumerate(dataset):

            # Forward Pass
            y_preds = model()


if __name__ == "__main__":

    training_text_path = "icdar2013-test-set/versions/4/Challenge2_Training_Task1_GT"
    training_img_path = "icdar2013-test-set/versions/4/Challenge2_Training_Task12_Images"

    training_data = load_data(text_path=training_text_path, image_path=training_img_path)

    training_data = DataLoader(training_data, batch_size=32, shuffle=32)

    train_features_batch, train_labels_batch = next(iter(training_data))

    print(train_features_batch[0].shape)
    exit()
    model = TextRecognitionModel(input)
    loss_fn = nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(params=model.parameters(), lr=0.01)