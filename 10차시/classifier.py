import torch.nn as nn
import torch

# 3계층 신경망
class AdvancedClassifier(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes, dropout_rate= 0.4):
        super(AdvancedClassifier, self).__init__()

        self.layer1= nn.Linear(input_size, hidden1_size)
        self.bn1= nn.BatchNorm1d(hidden1_size)
        self.relu1= nn.ReLU()
        self.dropout1= nn.Dropout(dropout_rate)

        self. layer2= nn.Linear(hidden1_size, hidden2_size)
        self.bn2= nn.BatchNorm1d(hidden2_size)
        self.relu2= nn.ReLU()
        self.dropout2= nn.Dropout(dropout_rate)

        self.layer3= nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        x= self.layer1(x)
        x= self.bn1(x)
        x= self.relu1(x)
        x= self.dropout1(x)

        x= self.layer2(x)
        x= self.bn2(x)
        x= self.relu2(x)
        x= self.dropout2(x)

        return self.layer3(x)
    
input_size= 64
output_size= 10

model= AdvancedClassifier(input_size, 128, 64, output_size)

x= torch.randn(2, input_size)
output= model(x)
print(f"입력 데이터: {x}")
print(f"출력 데이터: {output}")

# ------------------------------
# 모델 저장
# 위의 모델이 학습을 통해 parameter가 업데이트 되었다고 가정

MODEL_PATH= r'C:\Users\jinny\Desktop\두산로보틱스 부트캠프\AI\AI 개론 연습\10차시\model.pth'

# model의 state_dict 저장
# torch.save(저장할 객체, 경로)
torch.save(model.state_dict(), MODEL_PATH)

# 모델 불러오기
# 새로운 모델 instance 만들 때 처음에 선언한 구조와 동일한 인자를 넣어줘야 함
model_new= AdvancedClassifier(input_size, 128, 64, output_size)

# MODEL_PATH로부터 state_dict 불러오기
state_dict= torch.load(MODEL_PATH)

# model_new에 불러온 state_dict 적용
model_new.load_state_dict(state_dict)

model_new.eval()
print("새로운 모델 구조")
print(model_new)

# 두 모델의 parameter가 동일한지 확인
match= all(
    torch.equal(p1, p2)
    for p1, p2 in zip(model.parameters(), model_new.parameters())
)

print(f"두 모델의 parameter 일치 여부: {match}")