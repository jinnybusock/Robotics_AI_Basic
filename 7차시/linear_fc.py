import torch
import torch.nn as nn

layer= nn.Linear(3, 1)

# 가중치와 편향 초기화
nn.init.constant_(layer.weight, 2.0)
nn.init.constant_(layer.bias, 0.5)

print(f"가중치: {layer.weight.data}")
print(f"편향: {layer.bias.data}")