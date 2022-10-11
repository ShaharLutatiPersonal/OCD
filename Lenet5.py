import torch.nn as nn
from copy import deepcopy
# original paper - after each one need ReLU, we'll use relu cause cooler.
# conv2d - stride 1,kernel 5*5,size 28x28,channels 6
# average polling stride 2, kernel 2x2,size 14x14, channels 6
# conv2d stride 1, kernel 5x5,size 10x10,channels 16
# average polling stride 2, kernel 2x2 , size 5x5 ,channels 16
# conv2d stride 1 , kernel 5x5, size 1x1, channels 120
# FC1 size 84
# FC size 10


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

import torch
import torchvision
import torch.nn as nn


class Vgg_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg11(pretrained=True)
        self.linear = nn.Linear(1000,30)
        self.linear2 = nn.Linear(30,10)
        self.relu = nn.ReLU()
        self.vgg = vgg

    def forward(self, x):
        with torch.no_grad():
            out = self.vgg(x)
        out = self.relu(out)#.relu()
        latent_in  = deepcopy(out.detach())
        out = self.linear(out)
        out = self.relu(out)
        latent  = deepcopy(out.detach())
        out = self.linear2(out)
        out = out.sigmoid()
        return out,(latent,latent_in)


class NetD(nn.Module):
    def __init__(self,ch_in = 1 ):
        super(NetD, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 6, 5, padding=2)
        self.dropout1 = nn.Dropout2d(p=0.8)
        self.ReLU1 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(6)
        self.BN3 = nn.BatchNorm2d(16)
        self.BN4 = nn.BatchNorm2d(120)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.dropout2 = nn.Dropout2d(p=0.8)
        self.ReLU3 = nn.ReLU()
        self.fc1 = nn.Linear(480, 84)
        self.dropout3 = nn.Dropout(p=0.8)
        self.ReLU4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input):
        output = self.conv1(input)
        output = self.BN2(output)
        output = self.ReLU1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.BN3(output)
        output = self.ReLU2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.BN4(output)
        output = self.ReLU3(output)
        output = output.view(-1, self.num_flat_features(output))
        #output = self.dropout2(output)
        latent_in  = deepcopy(output.detach())
        output = self.fc1(output)
        output = self.dropout3(output)
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


class NetBN(nn.Module):
    def __init__(self):
        super(NetBN, self).__init__()
        self.BN1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.BN2 = nn.BatchNorm2d(6)
        self.ReLU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.BN3 = nn.BatchNorm2d(16)
        self.ReLU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.BN4 = nn.BatchNorm2d(120)
        self.ReLU3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.ReLU4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, input):
        output = self.BN1(input)
        output = self.conv1(output)
        output = self.BN2(output)
        output = self.ReLU1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.BN3(output)
        output = self.ReLU2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.BN4(output)
        output = self.ReLU3(output)
        output = output.view(-1, self.num_flat_features(output))
        output = self.fc1(output)
        output = self.ReLU4(output)
        output = self.fc2(output)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
