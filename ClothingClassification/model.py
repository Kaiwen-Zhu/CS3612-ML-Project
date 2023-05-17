import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(x.shape)    
        x = self.relu(self.bn1(self.conv1(x)))
        # print(x.shape)    
        x = self.relu(self.bn2(self.conv2(x)))
        # print(x.shape)    
        x = self.pool(x)
        # print(x.shape)    
        x = self.relu(self.bn3(self.conv3(x)))
        # print(x.shape)    
        x = self.relu(self.bn4(self.conv4(x)))
        # print(x.shape)    
        x = self.pool(x)
        # print(x.shape)    
        x = x.view(-1, 64*8*8)
        # print(x.shape)    
        x = self.dropout(self.relu(self.bn5(self.fc1(x))))
        # print(x.shape)    
        x = self.log_softmax(self.fc2(x))
        # print(x.shape)    
        return x