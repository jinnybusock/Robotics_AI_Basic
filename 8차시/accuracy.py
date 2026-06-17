import torch

outputs= torch.tensor([0.3, 0.5, 0.2, 0.9])
labels= torch.tensor([0.0, 1.0, 1.0, 1.0])

predicted= torch.where(outputs>0.5, 1, 0)

corrected= torch.where(predicted== labels, 1, 0)
corrected_predictions= corrected.sum()

accuracy= corrected_predictions/len(outputs)

print(f"예측: {predicted}")
print(f"맞춘 개수: {corrected_predictions}")
print(f"정확도: {accuracy}")