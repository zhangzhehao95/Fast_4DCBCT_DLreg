import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
#  Convolution block
#  conv (+ normaliation + relu)
# -------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, dim=3, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 mode='CIL'):
        super().__init__()
        blocks = []
        for t in mode:
            if t == 'C':
                Conv = getattr(nn, 'Conv%dd' % dim)
                blocks.append(Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias, padding_mode='replicate'))
            elif t == 'T':
                ConvTranspose = getattr(nn, 'ConvTranspose%dd' % dim)
                blocks.append(ConvTranspose(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, bias=bias, padding_mode='replicate'))
            elif t == 'B':
                BatchNorm = getattr(nn, 'BatchNorm%dd' % dim)
                blocks.append(BatchNorm(out_channels))
            elif t == 'I':
                InstanceNorm = getattr(nn, 'InstanceNorm%dd' % dim)
                blocks.append(InstanceNorm(out_channels))
            elif t == 'R':
                blocks.append(nn.ReLU(inplace=False))
            elif t == 'r':
                blocks.append(nn.ReLU(inplace=True))
            elif t == 'L':
                blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
            elif t == 'l':
                blocks.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            elif t == 'P':
                blocks.append(nn.PReLU())
            else:
                raise NotImplementedError('Undefined type: '.format(t))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block(x)
        return out


# Connect multiple convolution blocks
class CascadeConv(nn.Module):
    def __init__(self, dim=3, in_channels=64, out_channels=64, nb_conv=2, mode='CIL'):
        super().__init__()
        blocks = [ConvBlock(dim, in_channels, out_channels, mode=mode)]
        for _ in range(1, nb_conv):
            blocks.append(ConvBlock(dim, out_channels, out_channels, mode=mode))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.block(x)
        return out


# Down-sampling + multiple convs
class DownBlock(nn.Module):
    def __init__(self, dim=3, in_channels=64, out_channels=64, down='MaxPool', nb_conv=2, mode='CIL'):
        super().__init__()
        # Choose method to perform downsampling
        if down == 'StrideConv':
            self.downsampling = ConvBlock(dim, in_channels, in_channels, kernel_size=2, stride=2, padding=0, mode='C')
        elif down == 'MaxPool':
            MaxPool = getattr(nn, 'MaxPool%dd' % dim)
            self.downsampling = MaxPool(2)
        else:
            # If not StrideConv or MaxPool, directly use interpolate for down sampling
            self.downsampling = None
            self.dim = dim

        self.conv_block = CascadeConv(dim, in_channels, out_channels, nb_conv, mode)

    def forward(self, x):
        if self.downsampling:
            downsampled = self.downsampling(x)
        else:
            downsampled = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                        align_corners=True, recompute_scale_factor=False)

        out = self.conv_block(downsampled)
        return out


# Up-sampling (x1) + skip connection(x2) + multiple convs
class UpBlock(nn.Module):
    def __init__(self, dim=3, in_channels=[64, 64], out_channels=64, up='UpSample', nb_conv=2, mode='CIL'):
        super().__init__()
        if up == 'TransConv':
            TransConv = getattr(nn, 'ConvTranspose%dd' % dim)
            self.up = TransConv(in_channels[0], in_channels[1], kernel_size=2, stride=2)
        elif up == 'UpSample':
            Conv = getattr(nn, 'Conv%dd' % dim)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear' if dim == 2 else 'trilinear',
                                                align_corners=True),
                                    Conv(in_channels[0], in_channels[1], kernel_size=1))
        else:
            # If not TransConv or UpSample, use nearest interpolate for up sampling
            Conv = getattr(nn, 'Conv%dd' % dim)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                    Conv(in_channels[0], in_channels[1], kernel_size=1))

        self.conv_block = CascadeConv(dim, 2*in_channels[1], out_channels, nb_conv, mode)

    def forward(self, x1, x2):
        upsampled = self.up(x1)
        skip = torch.cat([upsampled, x2], 1)    # Data: (B,C,D,H,W)
        out = self.conv_block(skip)
        return out


# Residual block
class ResBlock(nn.Module):
    def __init__(self, dim, in_channels=32, out_channels=64, bottleneck=False):
        super().__init__()
        # Whether use bottleneck for conv
        self.bottleneck = bottleneck
        if self.bottleneck:
            self.conv1 = ConvBlock(dim, in_channels, in_channels, kernel_size=1, padding=0,  mode='CIL')
            self.conv2 = ConvBlock(dim, in_channels, in_channels, mode='CIL')
            self.conv3 = ConvBlock(dim, in_channels, out_channels, kernel_size=1, padding=0, mode='CI')
        else:
            self.conv1 = ConvBlock(dim, in_channels, out_channels, mode='CIL')
            self.conv2 = ConvBlock(dim, out_channels, out_channels, mode='CI')

        # Skip connection within the residual block
        if in_channels == out_channels:
            self.conv_skip = None
        else:
            self.conv_skip = ConvBlock(dim, in_channels, out_channels, kernel_size=1, padding=0, bias=False, mode='CI')

    def forward(self, x):
        shortcut = x

        if self.bottleneck:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        if self.conv_skip:
            shortcut = self.conv_skip(shortcut)

        x = torch.add(x, shortcut)
        x = F.leaky_relu(x, negative_slope=0.2)
        return x


# Down-sampling + residual block
class DownBlock_residual(nn.Module):
    def __init__(self, dim=3, in_channels=64, out_channels=64, down='MaxPool', bottleneck=False):
        super().__init__()
        if down == 'StrideConv':
            self.downsampling = ConvBlock(dim, in_channels, in_channels, kernel_size=2, stride=2, padding=0, mode='C')
        elif down == 'MaxPool':
            MaxPool = getattr(nn, 'MaxPool%dd' % dim)
            self.downsampling = MaxPool(2)
        else:
            self.downsampling = None
            self.dim = dim

        self.conv_block = ResBlock(dim, in_channels, out_channels, bottleneck)

    def forward(self, x):
        if self.downsampling:
            downsampled = self.downsampling(x)
        else:
            downsampled = F.interpolate(x, scale_factor=0.5, mode='bilinear' if self.dim == 2 else 'trilinear',
                                        align_corners=True, recompute_scale_factor=False)

        out = self.conv_block(downsampled)
        return out


# Up-sampling (x1) + skip connection(x2) + residual block
class UpBlock_residual(nn.Module):
    def __init__(self, dim=3, in_channels=[64, 64], out_channels=64, up='UpSample', bottleneck=False):
        super().__init__()
        if up == 'TransConv':
            TransConv = getattr(nn, 'ConvTranspose%dd' % dim)
            self.up = TransConv(in_channels[0], in_channels[1], kernel_size=2, stride=2)
        elif up == 'UpSample':
            Conv = getattr(nn, 'Conv%dd' % dim)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear' if dim == 2 else 'trilinear',
                                                align_corners=True),
                                    Conv(in_channels[0], in_channels[1], kernel_size=1))
        else:
            # If not TransConv or UpSample, use nearest interpolate for up sampling
            Conv = getattr(nn, 'Conv%dd' % dim)
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                    Conv(in_channels[0], in_channels[1], kernel_size=1))

        self.conv_block = ResBlock(dim, 2*in_channels[1], out_channels, bottleneck)

    def forward(self, x1, x2):
        upsampled = self.up(x1)
        skip = torch.cat([upsampled, x2], 1)    # Data: (B,C,D,H,W)
        out = self.conv_block(skip)
        return out
