import torch
from torch import nn
from torch.utils.data import DataLoader

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt

class FashtionMnistModel(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):

        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feastures=input_shape, out_features=hidden_units),
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

if __name__ == "__main__":

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

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=32)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    print(train_dataloader)

    model0 = FashtionMnistModel(input_shape=784, hidden_units=10, output_shape=len(class_names))

    from helper_functions import accuracy_fn
    from timeit import default_timer as timer

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model0.parameters(), lr=0.1)