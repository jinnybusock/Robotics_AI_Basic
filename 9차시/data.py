import torch
import torch.nn as nn
# scikit-learn library
# iris: 연습용으로 내장해 둔 붓꽃 데이터셋
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MultiClassNet(nn.Module):
    def __init__(self, input_layer, output_layer):
        super().__init__()

        self.linear= nn.Linear(input_layer, output_layer)

    def forward(self, x):
        return self.linear(x)
    
# data load 및 분할
iris= load_iris()
x_data, y_data= iris.data, iris.target
x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size= 0.5, random_state= 123)

# 훈련용 입력 데이터
inputs= torch.tensor(x_train, dtype= torch.float32)
# 훈련용 정답 데이터
labels= torch.tensor(y_train, dtype= torch.long)
# 검증용 입력 데이터
inputs_test= torch.tensor(x_test, dtype= torch.float32)
# 검증용 정답 데이터
labels_test= torch.tensor(y_test, dtype= torch.long)

print(f"inputs data: {inputs}")
print(f"labels data: {labels}")
print(f"labels_test data: {labels_test}")