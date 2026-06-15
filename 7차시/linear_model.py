import torch
import torch.nn as nn

class MyRegressionModel(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.layer1= nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.layer1(x)
    
net= MyRegressionModel(5, 1)
print(net)

dummy_input= torch.randn(2, 5)
output= net(dummy_input)

print(f"입력 값: {dummy_input.data}")
print(f"출력 값: {output.data}")