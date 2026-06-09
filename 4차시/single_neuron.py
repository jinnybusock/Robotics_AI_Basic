import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# 손실함수 정의
def mse(Yp, Y):
    loss= ((Yp-Y)** 2).mean()

    return loss

# 예시 데이터 생성(무작위)
X= torch.randn(5).float()
Y= torch.randn(5).float()
print(f"입력 데이터: {X}")
print(f"정답: {Y}")

# 초기 파라미터 및 학습률
W= torch.tensor(1.0, requires_grad= True).float()
B= torch.tensor(1.0, requires_grad= True).float()
print(f"초기 W: {W}")
print(f"초기 B: {B}")

lr= 0.01

# 반복 횟수
epochs= 500

# 예측 함수 정의
def pred(x):
    return W* x+ B

# 기록을 위한 배열 초기화 --> 그래프 그릴 때 활용
history= np.zeros((0, 2))

# 뉴런 학습 반복 처리
for epoch in range(epochs):
    # 예측 계산
    Yp= pred(X)

    # 손실함수 계산
    loss= mse(Yp, Y)

    # 경사 계산
    loss.backward()

    # 경사 기반으로 parameter 수정
    with torch.no_grad():   # 해당 블록 내에서는 계산 그래프 생성이 일시적 중단
        W-= lr* W.grad
        B-= lr* B.grad

    # 경사값 초기화
    W.grad.zero_()
    B.grad.zero_()

    # 추가 설명 참고
    if epoch% 10== 0:
        item= np.array([epoch, loss.item()])
        history= np.vstack((history, item))

print(f"업데이트 후 W: {W}")
print(f"업데이트 후 B: {B}")
print(f"W의 현재 경사: {W.grad}")
print(f"B의 현재 경사: {B.grad}")

# ------------------------------------

# 손실 그래프 출력
# x축: history의 모든 행의 첫 번째 열(반복 횟수)
# y축: history의 모든 행의 두 번째 열(손실)
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel("epoch")
plt.ylabel("loss function")
plt.show()