import torch
import torch.nn as nn
import torch.optim as optim
from binary_classifier import MyBinaryClassifier

input_layer= 2
output_layer= 1
learning_rate= 0.01

net= MyBinaryClassifier(input_layer, output_layer)
criterion= nn.BCELoss()
optimizer= optim.SGD(net.parameters(), lr= learning_rate)