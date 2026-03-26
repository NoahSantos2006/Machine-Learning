import torch

"""

Pattern 1: Direct Creation From Data

    You have a python list, and you want a tensor. Use torch.tensor()

"""

data = [[1, 2, 3], [4, 5, 6]]
my_tensor = torch.tensor(data)
# print(my_tensor)

"""

Pattern 2: Creation From A Desired Shape

    You know the shape you need, but not the values yet. This is how you initialize model weights

"""

# Input: A shape tuple (2 rows, 3 columns)
shape = (2,3)

ones = torch.ones(shape) # Initializes to all 1s
zeros = torch.zeros(shape) # Initializes to all 0s
random = torch.randn(shape)
# print(F"Random Tensor:\n {random}")

"""

Pattern 3: Creation By Mimicking Another Tensor

    You need a new tensor with the exact same shape and type as another one

"""

# Input: A 'template' tensor
template = torch.tensor([[1, 2], [3, 4]])

# Create a new tensor with the same properties
rand_like = torch.randn_like(template, dtype=torch.float)

print(f"Template Tensor:\n {template}\n")
print(f"Randn_like Tensor:\n {rand_like}")