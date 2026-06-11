import torch
import torch.nn as nn
import torch.optim as optim

x= torch.randn(8, 10)
y= torch.tensor([0,1,2,1,0,2,1,0])
print(f"입력 데이터: {x.data}")

model= nn.Linear(10, 3)

# label smoothing
smoothing_alpha= 0.1
criterion= nn.CrossEntropyLoss(label_smoothing= smoothing_alpha)

optimizer= optim.SGD(model.parameters(), lr= 0.1)

logits= model(x)

loss= criterion(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"손실 값: {loss.item():.4f}")