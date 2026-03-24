import torch
"""

Autograd: Forward and Backward

You define the forward pass; PyTorch takes care of the gradient computation
 - Build a computation graph from input to output (the forward pass)
 - PyTorch handles asuto differentiation and the backward pass
 - Resulting gradient tensor matches the input's shape
 - You just create the forward; PyTorch computs and lets you optimize


Gradient Accumulation

Gradients are not overwritten -- they are added up (or accumulated) on the x.grad attribute
 - lets you sum gradients across multiple forward passes
 - Useful for effective larger batch sizes or gradient accumulation steps
 - This is why you can use multiple loss functions to train your neural networks 


"""


x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward pass: compute z = x^2 + y^3
z = x**2 + y**3
print(f"z = {x}^2 + {y}^3 = {z}")

# Backward pass: compute gradients
z.backward() # automatically calculate gradient
dz_dx = x.grad # partial derivative wrt x
dz_dy = y.grad # partial derivative wrt y
print(f"dz/dx = {dz_dx}\ndz/dy = {dz_dy}\n")
