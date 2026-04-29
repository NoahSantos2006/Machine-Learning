import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):

        def __init__(self):

            super().__init__()

            self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

            self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        def forward(self, x: torch.Tensor) -> torch.Tensor:

            return self.weights * x * self.bias
        
if __name__ == "__main__":

    # Create *known* parameters
    weight = 0.7
    bias = 0.3

    # Create data
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    # Create train/test split
    train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):

        plt.figure(figsize=(10, 7))

        plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

        plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

        if predictions is not None:

            plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
        
        plt.legend(prop={"size":14})

        plt.show()

    torch.manual_seed(42)

    model_0 = LinearRegressionModel()

    with torch.inference_mode():
         y_preds = model_0(X_test)
    
    # Check the predictions
    print(f"Number of testing samples: {len(X_test)}") 
    print(f"Number of predictions made: {len(y_preds)}")
    print(f"Predicted values:\n{y_preds}")

    loss_fn = nn.L1Loss()

    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

    torch.manual_seed(42)

    epochs = 100

    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(epochs):
        ### Training
        
        # Put model in training mode (this is the default state of a model)
        model_0.train()

        # 1. Forward pass on train data using the forward() method inside
        y_pred = model_0(X_train)

        # 2. Calculate the loss (how different are out models predictions to the ground truth)
        loss = loss_fn(y_pred, y_train)

        # 3. Zero grad of the optimizer
        # We only want gradients from the current batch, not a sum of all the previous ones.
        # If we don't zero them then :
        #   - gradients get bigger
        #   - updates become incorrect
        #   - training can blow up or behave unpredictably
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Progress the optimizer
        optimizer.step()

        ### Testing

        # Put the model in evaluation mode
        model_0.eval()

        with torch.inference_mode():
                
            #1. Forward pass on test data
            test_pred = model_0(X_test)

            #2. Calculate loss on test data
            test_loss = loss_fn(test_pred, y_test.type(torch.float))

            # Print out what's happening
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")


