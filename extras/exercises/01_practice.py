import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class LinearRegressionModel(nn.Module):

    def __init__(self):

        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))

    def forward(self, x):
        return self.weight * x + self.bias

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight=0.3
    bias=0.9

    X = torch.arange(0, 1, 0.01)
    y = weight * X + bias

    train_split = int(len(X) * 0.8)

    X_train, y_train, X_test, y_test = X[:train_split], y[:train_split], X[train_split:], y[train_split:]

    def plot_data(train_data=X_train, train_label=y_train, test_data=X_test, test_label=y_test, predictions=None):

        plt.figure(figsize=(10, 7))

        plt.scatter(train_data, train_label, s=0.8, c="r", label="Train Data")
        plt.scatter(test_data, test_label, s=0.8, c="b", label="Test Data")

        if not predictions is None:

            plt.scatter(predictions, test_label, s=0.8, c="g", label="Predictions")
        
        plt.show()
    
    model = LinearRegressionModel()
    model.to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    epochs = 9999

    for epoch in range(epochs):

        # Put model into training mode
        model.train()

        # Forward Pass
        y_preds = model(X_train)

        # Calculate Loss
        loss = loss_fn(y_preds, y_train)

        # Zero optimizer
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Step optimizer
        # Updates model's parameters
        # Uses gradients computed in backward() and applies algorithm (SGD, Adam)
        optimizer.step()

        if epoch % 20 == 0:

            model.eval()

            test_preds = model(X_test)

            test_loss = loss_fn(test_preds, y_test)

            print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test Loss: {test_loss:.3f}")
        

    model.eval()

    with torch.inference_mode():

        y_preds = model(X_test)

        plot_data(predictions=y_preds)

    # SAVING MODELS
    
    # MODEL_PATH = Path("model")
    # MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # MODEL_NAME = "01 model"
    # MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)