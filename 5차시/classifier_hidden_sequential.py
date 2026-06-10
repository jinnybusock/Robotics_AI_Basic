import torch
import torch.nn as nn

class ClassifierwithSeq(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierwithSeq, self).__init__()

        self.network= nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
    
model= ClassifierwithSeq(20, 1)
print(model)

inputs= torch.randn(3, 20)
outputs= model(inputs)

print(f"입력 데이터: {inputs.data}")
print(f"출력 데이터: {outputs.data}")