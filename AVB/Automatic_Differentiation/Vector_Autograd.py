import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Forward pass: compute z = x^2 + y^3
z0 = (x**2 + y**3)
z = z0.sum()
print(f"z0 = x^2 + y^3 = {z0}\nz = {z}")

z.backward()
dz_dx = x.grad
dz_dy = y.grad
print(f"dz_dx = {dz_dx}\ndz/dy = {dz_dy}")