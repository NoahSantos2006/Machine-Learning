import torch

"""

Element-Wise Multiplication ('*')

    Multiplies matching positions
    Tensors must have the exact same shape

"""

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])

# This calculates: [[1*10, 2*20], [3*30, 4*40]]
element_wise_product = a * b

"""

Matrix Multiplication ('@')

    Powers neural networks
    When you build a linear layer, y = wx + b

"""

# Shape is (2, 3)
m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Shape is [3, 2]
m2 = torch.tensor([[7, 8], [9, 10], [11, 12]])

# Resulting shape will be: (2, 2) 
matrix_product = m1 @ m2