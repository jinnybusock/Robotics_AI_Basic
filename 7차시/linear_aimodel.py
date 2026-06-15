import torch
import torch.nn as nn
import torch.optim as optim

input_layer= 1
output_layer= 1
learning_rate= 0.01
epochs_nums= 500

class Net(nn.Module):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        self.layer1= nn.Linear(input_layer, output_layer)

    def forward(self, x):
        return self.layer1(x)
    
net= Net(input_layer, output_layer)

# 손실함수
criterion= nn.MSELoss()

# 최적화 함수
optimizer= optim.SGD(net.parameters(), lr= learning_rate)

inputs= torch.randn((10, 1))
answer= torch.randn((10, 1))

for epoch in range(epochs_nums):
    optimizer.zero_grad()   # 경사값 초기화
    outputs= net(inputs)

    # MSE 제곱 계산을 미분하면 앞에 상수 2가 붙기 때문에 2로 나누어줘야 함
    loss= criterion(outputs, answer)/ 2.0   # 손실 계산
    loss.backward()   # 경사 계산

    optimizer.step()   # 파라미터 수정

print(f"입력 데이터: {inputs.data}")
print(f"정답 데이터: {answer.data}")
print(f"계산된 손실 값: {loss.item():.4f}")