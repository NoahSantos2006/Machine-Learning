import torch
from torch import nn
from torch.utils.data import DataLoader

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

from helper_functions import accuracy_fn

class FashtionMnistModelV0(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):

        return self.layer_stack(x)

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():

        for X, y in data_loader:

            # 1. Forward Pass
            y_preds = model.eval(X)

            # 2. Calculate Loss (accumulate for the batch)
            loss += loss_fn(y_preds, y)
            acc += accuracy_fn(y_true=y, y_pred=y_preds.argmax(dim=1))

        
    # Find average loss per batch
    loss /= len(data_loader)
    acc /= len(data_loader)

    print(f"Test Loss: {loss:.5f} | Test Accuracy: {acc:.2f}%")

# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup training data
    train_data = datasets.FashionMNIST(
        root="data", # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
    )

    # Setup testing data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False, # get test data
        download=True,
        transform=ToTensor()
    )

    class_names = train_data.classes

    # Use DataLoader() to break the data into batches
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=32)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    train_features_batch, train_labels_batch = next(iter(train_dataloader))

    model0 = FashtionMnistModel(input_shape=784, hidden_units=10, output_shape=len(class_names))

    from helper_functions import accuracy_fn
    from timeit import default_timer as timer

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.1)

    from tqdm.auto import tqdm

    torch.manual_seed(42)
    train_time_start_on_cpu = timer()

    epochs = 3

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")

        train_loss = 0

        for batch, (X, y) in enumerate(train_dataloader):

            model0.train()

            # Forward Pass
            y_pred = model0(X)

            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            train_loss += loss

            # 3. Optimizer Zero grad
            optimizer.zero_grad()

            # 4. Backpropagate
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")
        
        # Divide total train loss by length of train dataloader (average loss per batch per epoch)
        train_loss /= len(train_dataloader)


        ### Testing
        # Setup variables for accumulatively adding up loss and accuracy
        test_loss, test_acc = 0, 0
        model0.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:

                # 1. Forward Pass
                test_pred = model0(X)

                # 2. Calculate loss (accumatively)
                test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

                # 3. Calculate accuracy (preds need to be same as y_true)
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            
            # Calculations on test  metrics need to happen inside torch.inference_mode()
            # Divide total test loss by length of test dataloader (per batch)
            test_loss /= len(test_dataloader)

            # Divide total accuracy by length of test dataloader (per batch)
            test_acc /= len(test_dataloader)
        
        print(f"\nTrain loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%\n")
    
    # Calculate training time
    train_time_end_on_cpi = timer()
    total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                                end=train_time_end_on_cpi,
                                                device=str(next(model0.parameters()).device))
    
    torch.manual_seed(42)
    model_2 = FashionMNISTModelV2(input_shape=1, 
        hidden_units=10, 
        output_shape=len(class_names)).to(device)



    """

    nn.Conv2d() layer expected 4-dimensional tensor as input [batch_size, color_channels, height, width]
    
    """

