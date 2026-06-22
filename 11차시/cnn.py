import torch.nn as nn
import torch

in_channels= 1
out_channels= 10
kernel_size= 5
pool_size= 2
stride_size= 2

conv1= nn.Conv2d(
    in_channels= in_channels,
    out_channels= out_channels,
    kernel_size= kernel_size
)

relu= nn.ReLU()

pool1= nn.MaxPool2d(
    kernel_size= pool_size,
    stride= stride_size
)

x= torch.randn(100, 1, 28, 28)
y1= conv1(x)
y2= relu(y1)
y3= pool1(y2)

print(f"원본 데이터 형태: {x.shape}")
print(f"CNN 통과 후: {y3.shape}")

# ------------------------------------
# Linear 함수에 넣기 위해 flatten하기

linear1= nn.Linear(10*12*12, 50)
linear2= nn.Linear(50, 10)

y_flatten= y3.view(100, -1)
y4= linear1(y_flatten)
y5= relu(y4)
y6= linear2(y5)

print(f"Final Shape: {y6.shape}")