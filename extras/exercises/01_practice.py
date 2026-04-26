import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class LinearRegressionModel(nn.Module):

    def __init__(self):

        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=float))
    
    def forward(self, x):

        return self.weight * x * self.bias

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight=0.3
    bias=0.9

    X = torch.arange(0, 1, 0.01, dtype=float)
    y = weight * X + bias

    train_split = int(len(X) * 0.8)
    X_train, y_train, X_test, y_test = X[:train_split], y[:train_split], X[train_split:], y[train_split:]

    def plot_data(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):

        plt.figure(figsize=(10, 7))

        plt.scatter(train_data, train_labels, s=0.8, c="r", label="Train Data")
        plt.scatter(test_data, test_labels, s=0.8, c="b", label="Test Data")

        if not predictions is None:

            plt.scatter(test_data, predictions, s=0.8, c="g", label="Predictions")

        plt.show()

    model = LinearRegressionModel()
    
    epochs = 300
    lr = 0.01

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    for epoch in range(epochs):

        # Put model into training mode
        model.train()

        # Forward Pass
        y_preds = model(X_train)

        # Calculate the loss
        loss = loss_fn(y_preds, y_train)

        # Zero Gradients
        optimizer.zero_grad()

        # Perform backpropagation on the loss
        loss.backward()

        # Update optimizer
        optimizer.step()

        if epoch % 20 == 0:

            model.eval()

            test_pred = model(X_test)

            test_loss = loss_fn(test_pred, y_test)

            # print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test Loss: {test_loss:.3f}")
    
    model.eval()

    with torch.inference_mode():

        y_preds = model(X_test)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "01_linear_regression_model"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    loaded_model = LinearRegressionModel()

    loaded_model.state_dict(torch.load(MODEL_SAVE_PATH))

    loaded_model.to(device=device)

    loaded_model.eval()

    with torch.inference_mode():

        loaded_preds = loaded_model(X_test)