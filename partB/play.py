from models import DefaultModel
import torch

m = DefaultModel()
x = torch.randn(16,3,28,28)
y = m(x)
print(y.shape)