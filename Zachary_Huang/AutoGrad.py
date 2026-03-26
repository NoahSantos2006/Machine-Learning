import torch

"""

requires_grad=True

    "This is a parameter. From now on, track *every single operation* that happens to it."

"""

# Data vs Parameter
x_data = torch.tensor([[1., 2.], [3., 4.]])
w = torch.tensor([[1.0], [2.0]], requires_grad=True)
# print(f"Data tensor requires_grad: {x_data.requires_grad}")
# print(f"Data tensor requires_grad: {w.requires_grad}")

"""

Building the Graph

    Goal: Compute z = x * y, where y = a + b

"""

# Three parameter tensors
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)

y = a + b # First operation

z = x * y # Second Operation

# .grad_fn - points to the function that created it
print(f"grad_fn for z: {z.grad_fn}") # grad_fn for z: <MulBackward0 object at 0x000001F6E9DF9750>
print(f"grad_fn for y: {y.grad_fn}") # grad_fn for y: <AddBackward0 object at 0x000001F6E9DF9750>
print(f"grad_fn for a: {a.grad_fn}") # None