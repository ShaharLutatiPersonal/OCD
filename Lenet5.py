import torch.nn as nn
from copy import deepcopy


class NetOriginal(nn.Module):
    def __init__(self,ch_in = 1 ):
        super(NetOriginal, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 6, 5, padding=2)
        self.ReLU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.ReLU3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.ReLU4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input):
        latent = []
        output = self.conv1(input)
        output = self.ReLU1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.ReLU2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.ReLU3(output)
        output = output.view(-1, self.num_flat_features(output))
        latent_in  = deepcopy(output.detach())
        output = self.fc1(output)
        output = self.ReLU4(output)
        latent = deepcopy(output.detach())
        output = self.fc2(output)
        return output,(latent,latent_in)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
