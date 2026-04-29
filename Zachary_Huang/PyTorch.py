import torch
import torch.nn as nn
import torch.optim as optim

"""

NN.MODULE

    - Instruction bookley and the baseplate for our LEGOs
    - Provides a standard structure for how your bricks connect

The Model Blueprint: 'NN.MODULE'

    - Inherit from torch.nn.Module
    - __init__: Define layers
    - forward: Connect layers

TORCH.OPTIM

    - Skilled builder who knows exactly how to adjust all the bricks according to the instructions from the gradients

    
"""

class LinearRegressionModel(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        # In the constructor, we DEFINE the layers we'll use
        self.linear_layer = nn.Linear(in_features, out_features) # y_hat = xW + b

    
    def forward(self, x):
        
        # In the foward pass, we CONNECT the layers
        return self.linear_layer(x)
    
# Our batch of data will have 10 data points
N = 10
# Each data point has 1 input feature and 1 output value
D_in = 1
D_out = 1

# Create input data X
X = torch.randn(N, D_in)

# Create true target labels y by using the "true" W and b
# The "true" W is 2.0, the "true" b is 1.0
# Model will never see true_W OR true_b
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # Add a little noise


# Instantiate the model
model = LinearRegressionModel(in_features=1, out_features=1)
print(f"Model Architecture:\n{model}")

# Next, replace manual weight update

# HyperParameters
learning_rate = 0.01

# Create an Adam optimizer
# We pass model.parameters() to tell it which tensors to manage
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# We'll also grab a pre-built loss functino from torch.nn
loss_fn = nn.MSELoss() # Mean Squared Error Loss

epochs = 100

for epoch in range(epochs):

    ### FORWARD PASS ###
    y_hat = model(X)

    ### CALCULATE LOSS ###
    loss = loss_fn(y_hat, y_true)

    ### Three-Line Mantra ###
    # 1. Zero the gradients
    optimizer.zero_grad()
    # 2. Compute Gradients
    loss.backward()
    # 3. Update the parameters
    optimizer.step()

    # Optional: Print Progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}")
