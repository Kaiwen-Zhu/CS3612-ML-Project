import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """A residual block consisting of two convolutional layers 
    with batch normalization and ReLU activation. The output and
    input have the same number of channels.
    """    
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out


class MyNet(nn.Module):

    def __init__(self, num_res_blocks=2, num_channel_1=32, num_channel_2=64, hidden_dim_fc=128):
        super().__init__()
        self.layer1 = self.make_layer(1, num_channel_1, num_res_blocks)
        self.layer2 = self.make_layer(num_channel_1, num_channel_2, num_res_blocks)
        self.fc1 = nn.Linear(num_channel_2*8*8, hidden_dim_fc)
        self.fc2 = nn.Linear(hidden_dim_fc, 10)
        self.dropout = nn.Dropout(0.5)

    def make_layer(self, in_channels, out_channels, num_res_blocks):
        """Constructs a layer of residual blocks ending with max pooling."""   

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            *[ResBlock(out_channels) for _ in range(num_res_blocks)],
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        
    def forward(self, x, return_emd: bool = False):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)

        layer2_out = layer2_out.view(layer2_out.shape[0], -1)
        fc1_out = self.fc1(layer2_out)
        fc2_out = self.fc2(F.relu(fc1_out))
        
        out = F.log_softmax(fc2_out, dim=1)

        if return_emd:
            return out, fc2_out, fc1_out, layer2_out, layer1_out
        return out
