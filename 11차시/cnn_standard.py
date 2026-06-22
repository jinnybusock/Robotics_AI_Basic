import torch.nn as nn

class SelfCNN(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super().__init__()

        self.conv1= nn.Conv2d(1, 32, 3)
        self.relu= nn.ReLU()
        self.conv2= nn.Conv2d(32, 32, 3)
        self.pool= nn.MaxPool2d((2,2))
        self.flat= nn.Flatten()
        self.l1= nn.Linear(input_layer, hidden_layer)
        self.l2= nn.Linear(hidden_layer, output_layer)

        self.features= nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.pool
        )

        self.classify= nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        y1= self.features(x)
        y2= self.flat(y1)

        return self.classify(y2)