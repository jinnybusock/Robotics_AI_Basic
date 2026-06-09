import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

def mse(Yp, Y):
    loss= ((Yp-Y)** 2).mean()

    return loss

X= torch.randn(5).float()
Y= torch.randn(5).float()
print(f"입력 데이터: {X}")
print(f"정답: {Y}")

W= torch.tensor(1.0, requires_grad= True).float()
B= torch.tensor(1.0, requires_grad= True).float()
print(f"초기 W: {W}")
print(f"초기 B: {B}")

lr= 0.01
epochs= 500

# optimzer로 SGD(확률적 경사 하강법) 사용
optimizer= optim.SGD([W, B], lr= lr)

def pred(x):
    return W* x+ B

history1= np.zeros((0, 2))

for epoch in range(epochs):
    # 경사값 초기화
    optimizer.zero_grad()

    Yp= pred(X)
    loss= mse(Yp, Y)
    loss.backward()

    # parameter 수정
    # optimizer가 알아서 parameter들 업데이트함
    optimizer.step()

    if epoch% 10== 0:
        item= np.array([epoch, loss.item()])
        history1= np.vstack((history1, item))

W= torch.tensor(1.0, requires_grad= True).float()
B= torch.tensor(1.0, requires_grad= True).float()

# momentum SGD(확률적 경사 하강법) 사용
optimizer= optim.SGD([W, B], lr= lr, momentum= 0.9)

history2= np.zeros((0, 2))

for epoch in range(epochs):
    # 경사값 초기화
    optimizer.zero_grad()

    Yp= pred(X)
    loss= mse(Yp, Y)
    loss.backward()

    # parameter 수정
    # optimizer가 알아서 parameter들 업데이트함
    optimizer.step()

    if epoch% 10== 0:
        item= np.array([epoch, loss.item()])
        history2= np.vstack((history2, item))

plt.plot(history1[:,0], history1[:,1], 'b', label= "SGD")
plt.plot(history2[:,0], history2[:,1], 'r', label= "Momentum")
plt.xlabel("epoch")
plt.ylabel("loss function")
# 범례 표시
plt.legend()
plt.show()