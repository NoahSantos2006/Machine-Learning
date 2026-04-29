import torch
from sklearn.datasets import make_circles
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CircleModelV0(nn.Module):

    def __init__(self):

        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        return self.layer3(x)

def accuracy_fn(y_true, y_pred):

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples = 1000

    X, y = make_circles(n_samples,
                        noise=0.03,
                        random_state=42)

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CircleModelV0().to(device)
    untrained_preds = model(X_test.to(device))

    # BCELoss = no sigmoid built-in (expects probabilities, not logits)
    loss_fn = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss = sigmoid built-in (expects logits, not probabilities)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    y_logits = model(X_test.to(device))[:5]
    
    # change logits into probablities since we used BCEWithLogitsLoss()
    y_pred_probs = torch.sigmoid(y_logits)

    # change to binary for actual output
    y_preds = torch.round(y_pred_probs)

    torch.manual_seed(42)

    epochs = 1000

    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    for epoch in range(epochs):
        
        # Put model into training
        model.train()

        # Forward Pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Loss Function
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_train, y_pred)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        if epoch % 100 == 0:

            model.eval()

            with torch.inference_mode():
            
                y_test_logits = model(X_test).squeeze()

                test_loss = loss_fn(y_test_logits, y_test)
                y_test_preds = torch.round(torch.sigmoid(y_test_logits))

                test_acc = accuracy_fn(y_test, y_test_preds)

                print(f"Epoch: {epoch} | Accuracy: {test_acc:.2f} | Test Loss: {test_loss:.2f}")
    
    from helper_functions import plot_predictions, plot_decision_boundary

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)

    plt.show()

