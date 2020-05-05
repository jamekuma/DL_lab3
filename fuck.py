import torchvision.models.resnet
import torch

C = 4
H = 10
W = 10

a = torch.ones(2, H, W, C)
b = torch.Tensor([[[[1, 2, 3, 4]]], [[[1, 2, 3, 4]]]])
print(a * b)
print()
print(torch.mul(a, b))