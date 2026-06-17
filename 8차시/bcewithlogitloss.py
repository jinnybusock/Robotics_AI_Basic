import torch
import torch.nn as nn

class NetWithLogits(nn.Module):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        self.linear= nn.Linear(input_layer, output_layer)
        
    def forward(self, x):
        return self.linear(x)
    
net= NetWithLogits(2, 1)
inputs= torch.randn(5, 2)
label= torch.randn(5,1)

criterion= nn.BCEWithLogitsLoss()
y= net(inputs)

result= criterion(y, label)

print(f"입력 값: {inputs}")
print(f"출력 값: {result}")
print(f"실제 값: {label}")