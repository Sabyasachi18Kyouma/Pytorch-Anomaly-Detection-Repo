import torch

from models.simply_model import SimpleClassifier


# Create model
model = SimpleClassifier()


# Create fake batch
x = torch.randn(32, 100)


# Run forward pass
output = model(x)


print(output)

print(output.shape)