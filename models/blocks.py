"""A list of commonly used building blocks."""

from torch import nn


class Conv2dBnRelu(nn.Module):
    """A commonly used building block: Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, pooling=None,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pooling = pooling
        self.activation = activation

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.pooling is not None:
            x = self.pooling(x)
        return self.activation(x)


class LinearBnRelu(nn.Module):
    """A commonly used building block: FC -> BN -> ReLU"""

    def __init__(self, in_features, out_features, bias=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.linear(x)))
