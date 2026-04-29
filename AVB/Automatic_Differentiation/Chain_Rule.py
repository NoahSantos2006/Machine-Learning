import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = y + 1
w = z**2

dw_dz = torch.autograd.grad(outputs=w, inputs=z, retain_graph=True)[0]
dz_dy = torch.autograd.grad(outputs=z, inputs=y, retain_graph=True)[0]
dy_dx = torch.autograd.grad(outputs=y, inputs=x, retain_graph=True)[0]
print(f"dw_dz = {dw_dz}, dz_dy = {dz_dy}, dy_dx = {dy_dx}")

dw_dx = dw_dz * dz_dy * dy_dx
print(f"dw/dx = {dw_dz} * {dz_dy} * {dy_dx} = {dw_dx}")

w.backward()
print(f"dw/dx = {x.grad}")