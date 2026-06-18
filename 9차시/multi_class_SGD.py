import torch.optim as optim
import torch.nn as nn
from data import MultiClassNet

input_layer= 4
output_layer= 3
learning_rate= 0.01

net= MultiClassNet(input_layer, output_layer)
criterion= nn.CrossEntropyLoss()
optimizer= optim.SGD(net.parameters(), lr= learning_rate)