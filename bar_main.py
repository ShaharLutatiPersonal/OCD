import torch
from torchvision.models import efficientnet_b0
from blockselect import block_selector
import Lenet5
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from FMix.models import ResNet18



model = ResNet18()
model = torch.hub.load('ecs-vlc/FMix:master', 'preact_resnet18_cifar10_baseline' , pretrained=True)
model.load_state_dict((torch.load("./base_models/model_1_FMIX_fashion.pt", map_location="cpu")))
train_ds = FashionMNIST("fashdata", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
loss = torch.nn.CrossEntropyLoss()
my_block_calc = block_selector.block_entropy_calc(model, iter(train_loader), loss, device="cpu")
result = my_block_calc.forward()
print(result)
print(result)




