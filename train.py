import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

import math
import sys
import pickle

from statistics import mean
from torch.utils.data import DataLoader

from model import Network
from dataset import Ising
from utils import evaluate

DATA_PATH = "../Ising/data/"
EPOCHS = 30
BATCH_SIZE = 8
CUDA = torch.cuda.is_available()

transforms = T.Compose(
    [
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5)
    ]
)

train_set = Ising(DATA_PATH, transforms=transforms, train=True)
test_set = Ising(DATA_PATH, train=False)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True 
)

net = Network()
if CUDA:
    net.to(torch.device('cuda'))

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(EPOCHS):
    total_mse = 0
    avg_mse = 0
    counter = 0
    for img, targets in train_loader:
        counter += 1
        if CUDA:
            img = img.cuda()
            targets["T"] = targets["T"].cuda()
        output = net(img)
        loss = F.mse_loss(output.flatten(), targets["T"])

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse += loss.item()
        avg_mse += loss.item()
        if not counter % 10:
            avg_mse = avg_mse/10
            print("After {} iterations, Total MSE: {}, Avg. MSE: {}". format(counter, total_mse, avg_mse))

    print("\nEpoch number {}; Total MSE: {}; Avg. MSE: {}\n".format(epoch, total_mse, avg_mse/counter))


test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=True 
)

losses, targets, predictions = evaluate(net, test_loader, CUDA=CUDA)

print(mean(losses))

torch.save(net.state_dict(), "./cfg/net.pt")
torch.cuda.empty_cache()