import torch

a = torch.rand(2,3)
b = torch.rand(2,3)
# print(f"a = {a}\nb = {b}\n")


add = a + b
sub = a - b
mul = a * b
# print(f"add = {add}\nsub = {sub}\nmul = {mul}\n")


a_transpose = a.T
# print(f"Shapes --> a: {a.shape}, b: {b.shape}\n")


dot_AB = a @ b.T # Transpose b
dot_BA = a.T @ b # Transpose a
# print(f"dot_AB: {dot_AB.shape}, dot_BA: {dot_BA.shape}\n")

# Broadcasting - Allows PyTorch to perform operations on tensors with different shapes
a = torch.tensor([[1], [2], [3]])
b = torch.tensor([4, 5, 6])
# print(f"a = {a.shape}\nb = {b.shape}\n")

# Instead of throwing an error, pytorch stretches the smaller tensor to match the dimension of the larger one
c = a + b
# print(f"a + b (shape {c.shape}) = \n {c}")



# Batch Multplication
batch_size = 10
a = torch.randn(batch_size, 3, 4)
b = torch.randn(batch_size, 4, 5)
# print(f"shapes: a = {a.shape}, b = {b.shape}")

c = a @ b
# print(f"shape: {c.shape}")
 
# d has no "batch" dimension, it's 2d
d = torch.randn(4, 5)
# matrix with broadcasting (@ means matrix multiplication)
e = a @ d
# print(f"shape of result: {e.shape}")

# Tensor reshape
a = torch.randn(size=[2, 3, 4])
a_flat = a.reshape(-1)
# print(a.shape, " --> ", a_flat.shape)

a_transpose_1 = a.transpose(1, 0)
a_transpose_2 = a.transpose(1, 2)
# print(a_transpose_1.shape, ", ", a_transpose_2)

a_permute = a.permute([1, 2, 0])
# print(a.shape, " --> ", a_permute.shape)

# Expanding and Squeezing
a_expanded = a.unsqueeze(1)
a_squeezed = a_expanded.squeeze(1)
print(a.shape, "-->", a_expanded.shape, "-->", a_squeezed.shape)

