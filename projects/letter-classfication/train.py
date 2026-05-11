import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

from load_dataset import load_data

class LetterClassification(nn.Module):

    def __init__(self, input_shape: int, hidden_features: int, output_shape: int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_features*7*7, out_features=output_shape)
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)

        return x

# Calculate accuracy (a classification metric)
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

def train_step(model, train_data, epochs, loss_fn, optimizer, accuracy_fn):

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_data):

            model.train()

            # 1. Forward Pass
            y_preds = model(X)

            # 2. Calculate Loss and Accuracy
            loss = loss_fn(y_preds, y)
            train_loss += loss

            # 3. Zero Grad Optimizer
            optimizer.zero_grad()

            # 4. Backpropogate
            loss.backward()

            # 5. Update optimizer
            optimizer.step()

            if batch % 400 == 0:

                tqdm.write(f"Looked at Batch {batch}: {batch*len(X)}/{len(train_data.dataset)} samples")

        train_loss /= len(train_data)

        tqdm.write(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}")

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "LetterClassification"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def eval_model(model, test_data, accuracy_fn, loss_fn):

    model.eval()

    loss, acc = 0, 0

    with torch.inference_mode():

        for X, y in test_data:

            y_logits = model(X)
            loss += loss_fn(y_logits, y)

            y_preds = y_logits.argmax(dim=1)
            acc += accuracy_fn(y_true=y, y_pred=y_preds)

        loss /= len(test_data)
        acc /= len(test_data)

        print(f"Accuracy: {acc} | Loss: {loss}")

if __name__ == "__main__":

    train_data, test_data, output_shape = load_data()

    LetterClassificationModel = LetterClassification(input_shape=1, hidden_features=10, output_shape=output_shape)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=LetterClassificationModel.parameters(), lr=0.01)

    # train_step(model=LetterClassificationModel, train_data=train_data, epochs=5, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)

    LetterClassificationModel = LetterClassification(input_shape=1, hidden_features=10, output_shape=output_shape)

    LetterClassificationModel.load_state_dict(torch.load("models/LetterClassification"))

    eval_model(LetterClassificationModel, test_data, accuracy_fn, loss_fn)

