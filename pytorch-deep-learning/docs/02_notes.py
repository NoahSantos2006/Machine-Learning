import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

class Classification(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.layer1 = nn.Linear(in_features=20, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=10)
        self.layer4 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))

        return x

def acc_fn(test_preds, y_preds):

    acc = torch.eq(test_preds, y_preds).sum().item()

    return (acc / len(y_preds)) * 100

if __name__ == "__main__":

    X, y = make_classification(n_samples=1000)

    X = torch.tensor(X).type(torch.float32)
    y = torch.tensor(y).type(torch.float32)

    model = Classification()

    epochs = 10000

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    for epoch in range(epochs):

        # Put model in training
        model.train()

        # Forward Pass
        y_logits = model(X_train).squeeze()

        # Calculate Loss
        loss = loss_fn(y_logits, y_train)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backpropagate
        loss.backward()

        # Update optimizers
        optimizer.step()

        if epoch % 1000 == 0:

            model.eval()

            with torch.inference_mode():

                test_logits = model(X_test).squeeze()
                test_preds = torch.round(torch.sigmoid(test_logits))

                test_loss = loss_fn(test_logits, y_test)
                test_acc = acc_fn(test_preds, y_test)

                print(f"Epoch: {epoch} | Test Accuracy: {test_acc} | Loss: {loss} | Test Loss: {test_loss}")
    
