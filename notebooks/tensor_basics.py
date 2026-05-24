import numpy as np
import torch


# Create NumPy array
signals = np.random.normal(0, 1,(32, 100))

print("NumPy Array:")
print(signals)

print("\nNumPy Type:")
print(type(signals))


# Convert NumPy array to PyTorch tensor
signals_tensor = torch.tensor(signals, dtype=torch.float32)

print("\nPyTorch Tensor:")
print(signals_tensor)

print("\nTensor Type:")
print(type(signals_tensor))


# Print tensor shape
print("\nTensor Shape:")
print(signals_tensor.shape)