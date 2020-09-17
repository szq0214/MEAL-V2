"""ResNet-50 architecture pytorch model."""

from torch import nn
from torch.nn import functional as F


class ResNet50(nn.Module):
    LR_REGIME = [1, 140, 0.1, 141, 170, 0.01, 171, 200, 0.001]

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self._current_planes = 64
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(planes=64, num_layers=3)
        self.layer2 = self._make_layer(planes=128, num_layers=4, stride=2)
        self.layer3 = self._make_layer(planes=256, num_layers=6, stride=2)
        self.layer4 = self._make_layer(planes=512, num_layers=3, stride=2)
        self.fc = nn.Linear(512 * 4, 1000)

    def _make_layer(self, planes, num_layers, stride=1):
        layers = []
        # Add blocks one by one
        for i in range(num_layers):
            # Apply the stride on the first block of the series
            block_stride = stride if i == 0 else 1
            # If input size is changing, do convolution downsampling (residual)
            if block_stride != 1 or self._current_planes != planes * 4:
                downsample = nn.Sequential(
                    nn.Conv2d(self._current_planes, planes * 4, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(planes * 4))
            else:
                downsample = None
            layers.append(Bottleneck(
                self._current_planes, planes, stride=block_stride,
                downsample=downsample))
            self._current_planes = planes * 4
        # Make a sequential of all blocks
        return nn.Sequential(*layers)

    def classifier(self, x):
        return self.fc(x)

    def feats(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.feats(x)
        x = self.classifier(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            # Compute residual at the beginning so we can free the memory of x,
            # if not needed.
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)

        x += residual
        x = F.relu(x, inplace=True)
        return x
