import torch
import torch.nn as nn
import torch.optim as optim
from multi_class_SGD import net, criterion, optimizer
from data import inputs, labels, labels_test, inputs_test

# 모델 학습
epochs= 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs= net(inputs)

    loss= criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 검증 데이터로 예측
net.eval()   # 모델을 평가 모드로 전환
with torch.no_grad():
    test_ouputs= net(inputs_test)

# 정확도 계산
predicted= torch.max(test_ouputs, 1)[1]

correct= torch.where(predicted== labels_test, 1, 0).sum()

accuracy= correct/ len(labels_test)

print(f"예측 클래스: {predicted.tolist()[:10]}")
print(f"정답 클래스: {labels.tolist()[:10]}")
print(f"맞춘 건수: {correct.item()}")
print(f"정확도: {accuracy.item():.2f}")