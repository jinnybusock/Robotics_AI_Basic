import torch
import torch.nn as nn

class CustomMAELoss(nn.Module):
    def __init__(self):
        super(CustomMAELoss, self).__init__()

    def forward(self, yp, yt):
        return torch.abs(yp- yt).mean()
    
criterion= CustomMAELoss()
yp= torch.randn(3)
yt= torch.randn(3)
print(f"yp 값: {yp.data}")
print(f"yt 값: {yt.data}")

test_loss= criterion(yp, yt)
print(f"손실: {test_loss:.4f}")