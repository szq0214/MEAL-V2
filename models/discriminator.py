
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, outputs_size, K = 2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv2 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size//K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv3 = nn.Conv2d(in_channels=outputs_size, out_channels=2, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = x[:,:,None,None]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        return out

