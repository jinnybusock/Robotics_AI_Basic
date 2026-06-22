import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 28x28 이미지를 784차원 벡터로 변환
transform= transforms.Compose([
    # 텐서로 변환
    transforms.ToTensor(),
    # 1차원 벡터로 펼치기
    transforms.Lambda(lambda x: x.view(-1))
])

train_set= datasets.MNIST(
    root= './data',
    train= True,
    download= True,
    transform= transform
)

batch_size= 100

train_loader= DataLoader(
    train_set,
    batch_size= batch_size,
    shuffle= True
)

print(f"데이터셋 개수: {len(train_set)}")
print(f"데이터로더 배치 개수: {len(train_loader)}")