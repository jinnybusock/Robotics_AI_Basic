import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierModel, self).__init__()

        # 필요한 layer들 정의
        self.layer1= nn.Linear(input_dim, 64)
        self.relu1= nn.ReLU()
        self.layer2= nn.Linear(64, 32)
        self.relu2= nn.ReLU()
        self.layer3= nn.Linear(32, output_dim)
        self.sigmoid= nn.Sigmoid()

    def forward(self, x):
        l1= self.relu1(self.layer1(x))
        l2= self.relu2(self.layer2(l1))
        out= self.sigmoid(self.layer3(l2))

        return out
    
model= ClassifierModel(20, 1)
print(model)

inputs= torch.randn(3, 20)
outputs= model(inputs)

print(f"입력 값: {inputs.data}")
print(f"출력 값: {outputs.data}")