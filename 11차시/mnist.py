import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn_standard import SelfCNN
import torch.nn as nn
import torch.optim as optim

# Dataset Load
transforms_cnn= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])

train_set= datasets.MNIST(root='./data', train= True, download= True, transform= transforms_cnn)
test_set= datasets.MNIST(root= './data', train= False, download= True, transform= transforms_cnn)

batch_size= 128

train_loader= DataLoader(train_set, batch_size= batch_size, shuffle= True)
test_loader= DataLoader(test_set, batch_size= batch_size, shuffle= False)

# data 크기 보기
data_iter= iter(train_loader)
images, labels= next(data_iter)
print(f"입력 이미지 크기: {images.shape}")

net= SelfCNN(32*12*12, 128, 10)

# 모델 학습
criterion= nn.CrossEntropyLoss()
lr= 0.01
optimizer= optim.SGD(net.parameters(), lr= lr)
epochs= 5

for epoch in range(epochs):
    running_loss= 0.0

    for x, y in train_loader:
        optimizer.zero_grad()

        yp= net(x)
        loss= criterion(yp, y)

        loss.backward()

        optimizer.step()
        running_loss+= loss.item()

    # epoch마다 평균 loss 출력
    epoch_loss= running_loss/ len(train_loader)
    print(f"Epoch [{epoch}/{epochs}]- Loss: {epoch_loss}")