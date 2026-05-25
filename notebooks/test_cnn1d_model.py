import torch

from models.cnn1d_model import CNN1DClassifier


model = CNN1DClassifier()

x = torch.randn(32, 100)

output = model(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)