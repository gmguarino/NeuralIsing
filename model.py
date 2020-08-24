import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision


class Network(torch.nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)  
        
        # need to flatten in forward
        
        self.fc1 = nn.Linear(in_features=12*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)        
        self.fc3 = nn.Linear(in_features=60, out_features=16)   
        self.out = nn.Linear(in_features=16, out_features=1)   
    
    def forward(self, t):
        t = t # (1) input layer.. we would avoid this normally
        
        # (2) hidden conv layer
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (4) hidden linear layer
        t = t.reshape(t.size(0), -1)
        t = F.relu(self.fc1(t))
        
        # (5) hidden linear layer
        t = F.relu(self.fc2(t))

        # (5) hidden linear layer
        t = F.relu(self.fc3(t))
        
        # (6) output layer
        t = self.out(t)

        return t