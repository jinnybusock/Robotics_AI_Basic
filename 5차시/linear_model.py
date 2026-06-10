import torch
import torch.nn as nn

# 100개의 데이터
# 각 데이터는 784개의 요소를 가짐
inputs= torch.randn(100, 784)

# 첫 번째 linear 함수 통과
# 784개의 입력을 받아 128개의 출력
l1= nn.Linear(784, 128)
m1= l1(inputs)

# 활성화 함수 통과
relu= nn.ReLU(inplace= True)
m2= relu(m1)

# 두 번째 선형 함수 통과
l2= nn.Linear(128, 10)
outputs= l2(m2)

# 전체를 하나의 합성 함수로 정의
net= nn.Sequential(
    l1,
    relu,
    l2
)

result= net(inputs)