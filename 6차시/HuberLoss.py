import torch
import torch.nn as nn

def huber_loss(yp, yt, delta= 1.0):
    err= torch.abs(yp- yt)

    condition= (err> delta)

    loss_True= 1.0* (err- 0.5* 1.0)
    loss_False= 0.5* (yp- yt)**2

    loss= torch.where(condition, loss_True, loss_False)

    # 전체 배치의 평균 손실 반환
    return loss.mean()

criterion= huber_loss

yp= torch.randn(3)
yt= torch.randn(3)
print(f"yp 값: {yp.data}")
print(f"yt 값: {yt.data}")

test_loss= criterion(yp, yt)
print(f"손실 값: {test_loss.item():.3f}")