import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from src.models.utils import CartesianPositionalEmbedding

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LatentEncoder(ModelMixin, ConfigMixin):
    def __init__(
            self,
            in_channels =3,
            enc_channels=128,
            num_components=4,
            image_size=128,
            latent_dim = 64
    ):
        super().__init__()
        self.num_components = num_components
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.latent_dim_expand =self.latent_dim * self.num_components
        kernel_size = 3
        encode_depth = 3
        out_dim = self.latent_dim_expand

        reduced_size = image_size
        for i in range(encode_depth):
            reduced_size = (1 + reduced_size) // 2  # halved (ceiling) after each conv
        activ = nn.ReLU
        layers = [nn.Conv2d(in_channels, enc_channels, kernel_size=kernel_size, stride=1, padding=1), activ()]
        enc = enc_channels

        for i in range(encode_depth):
            layers.append(BasicBlock(enc, enc))
            layers.append(activ())
            layers.append(nn.Conv2d(enc, 2 * enc, kernel_size, stride=2, padding=1))
            layers.append(activ())
            enc *= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(enc * reduced_size * reduced_size, out_dim))
        layers.append(activ())  # can experiment with removing

        self.latent_encoder = nn.Sequential(*layers)

    def forward(self, x):
        slots = self.latent_encoder(x)
        slots = slots.view(slots.shape[0], self.num_components, self.latent_dim)
        return slots