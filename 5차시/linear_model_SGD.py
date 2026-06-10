import torch
import torch.optim as optim
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear= nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
X= torch.randn(100, 1)
Y= 2* X+ 1+ 0.1* torch.randn(100, 1)

learningRate= 0.01
epochs= 100

model= LinearModel(1, 1)

# 손실함수 정의
criterion= nn.MSELoss()

# 최적화 함수 정의
optimizer= optim.SGD(model.parameters(), lr= learningRate)

for epoch in range(epochs):
    Yp= model(X)
    loss= criterion(Yp, Y)

    # 경사 초기화
    optimizer.zero_grad()

    # 역전파 계산
    loss.backward()

    # 파라미터 업데이트
    optimizer.step()

    if epoch% 10== 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

[W, B]= model.parameters()
print(f"학습된 W: {W.item():.4f}")
print(f"학습된 B: {B.item():.4f}")