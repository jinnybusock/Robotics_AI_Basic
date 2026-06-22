import torch.nn as nn
import torch

class Net_FCNN(nn.Module):
    def __init__(self, input_layer, output_layer, hidden_layer):
        super().__init__()

        self.l1= nn.Linear(input_layer, hidden_layer)
        self.l2= nn.Linear(hidden_layer, output_layer)

        self.relu= nn.ReLU()

    def forward(self, x):
        x= self.l1(x)
        x= self.relu(x)

        return self.l2(x)
    
input_layer= 784
output_layer= 10
hidden_layer= 128

net= Net_FCNN(input_layer, output_layer, hidden_layer)
x= torch.randn(100, 784)
output= net(x)
print(f"Model Output: {output}")