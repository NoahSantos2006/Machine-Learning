import torch

tensor = torch.randn(2, 3)

# .shape: A tuple describing the dimensions. 90% of errors come from shape
print(f"Shape: {tensor.shape}")

# .device: Where the tensor lives. cpu or cuda (GPU)
print(f"SDatatype: {tensor.dtype}")

# .dtype: The data type of the numbers. Default is float32 (due to gradients)
print(f"Device: {tensor.device}")