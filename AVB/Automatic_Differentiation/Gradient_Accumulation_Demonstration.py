import torch

x = torch.tensor(1.0, requires_grad=True)
# First computation
y1 = x**2
y1.backward()
print(F"After first backward: x.grad = {x.grad}")

# Zero gradients
x.grad.zero_()
print(f"After zeroing: x.grad = {x.grad}")

# Second computation (gradients accumulate)
y2 = x**3
y2.backward()
print(f"After second backward: x.grad = {x.grad}") # 2*x + 3*x^2 = 2 + 3