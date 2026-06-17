import torch
import torch.nn as nn

class MyBinaryClassifier(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.linear= nn.Linear(n_input, n_output)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        y= self.linear(x)

        return self.sigmoid(y)
    
net= MyBinaryClassifier(2, 1)
print(net)

x= torch.randn(2, 2)
y= net(x)

print(f"입력 값: {x}")
print(f"출력 값: {y}")