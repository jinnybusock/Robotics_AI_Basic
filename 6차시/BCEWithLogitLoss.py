import torch
import torch.nn as nn
import torch.optim as optim

x= torch.randn(10, 5)
# view 추가 설명 참고
y= torch.tensor([1,0,1,0,1,1,0,0,1,1]).view(-1, 1).float()

model= nn.Linear(5, 1)
criterion= nn.BCEWithLogitsLoss()
optimizer= optim.SGD(model.parameters(), lr= 0.1)

logits= model(x)
# logits 값이 소수이기 때문에 y 값도 소수로 맞춰줘야 함
loss= criterion(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"손실 계산: {loss.item():.3f}")