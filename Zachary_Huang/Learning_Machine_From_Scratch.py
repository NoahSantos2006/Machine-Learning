import torch

"""

Forward Pass: Model's First Guess (random)

The Model: Simple Linear Regression (y = XW + b)

    - X = input data
    - W = weight
    - b = bias
    - y (y-hat) = model's prediction

"""

# Setup: Creating Data

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

# Initialize our parameters with random values
# Shapes must be correct for matrix multiplication
W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Initial Weight W:\n {W}\n")
print(f"Initial Bias b:\n {b}\n")

y_hat = X @ W + b

print(f"Prediction y_hat (first 3 rows):\n {y_hat[:3]}\n")
print(f"True Labels y_true (first 3 rows):\n {y_true[:3]}\n")

# Calculate loss
error = y_hat - y_true
squared_error = error ** 2
loss = squared_error.mean()

# Loss sits at end of computation graph
# Source from which all knowledge will flow backward
print(f"Loss (our single scorecard number): {loss}\n")

# .backward() : Travel backward from 'loss' and calcaulte gradients for all parameters with 'requires_grad=True'
# e.g. gradient of the Loss w.r.t our Weight 'w'

# Compuite gradients
loss.backward()

# The gradients are now stored in the .grad attribute

#   Negative gradient --> increasing W decreases loss.
#   Gradient points toward steepest increase
#   Go opposite direction to minimuze loss

print(f"Gradient for W:\n {W.grad}\n")
print(f"Gradient for b:\n {b.grad}\n")

"""

We have what we need to improve

    - Measure error: loss
    - Know direction: .grad

"""

# Training Loop (gradient descent --> x_next = x_current - learning_rate * gradient in loss)
# W_new = W_old - learning rate * W.grad
# b_new = b_old - learning_rate * b.grad

"""

The Training Loop

    - Repeat out 5 steps for multiple epochs
    - torch.no_grad(): Don't track parameter updates
    - .grad.zero_(): Reset gradients each iteration

"""

# Hyperparameters
learning_rate, epochs = 0.01, 100

# Re-initialize parameters
W, b = torch.randn(1, 1, requires_grad=True), torch.randn(1, requires_grad=True)

# Training loop
for epoch in range(epochs):
    # Forward pass and loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true)**2) # MSE

    # Backward pass
    loss.backward()

    # Update parameters
    # We do torch.no_grad() here so the manual change of weights is not tracked on the computation graph
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
    
    # Zero gradients (reset for next round of learning)
    W.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0:

        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}, W={W.item():.3f}, b={b.item():.3f}")

print(f"\nFinal Parameters: W={W.item():.3f}, b={b.item():.3f}")
print(f"True Parameters: W=2.000, b=1.000")