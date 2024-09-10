import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from mamba_ssm import Mamba



class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_mamba_layer(
    in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return mamba_layer


class RMB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = nn.BatchNorm2d(in_channels,device='cuda:0')  # Assuming spatial_dims is 2 for 2D convolution
        self.norm2 = nn.BatchNorm2d(in_channels,device='cuda:0')  # Assuming spatial_dims is 2 for 2D convolution
        self.act = nn.ReLU(inplace=True)  # Assuming act is ReLU
        self.conv1 = MambaLayer(
            input_dim=in_channels, output_dim=in_channels
        )
        self.conv2 = MambaLayer(
            input_dim=in_channels, output_dim=in_channels
        )
        self.device = "cuda:0"

    def forward(self, x):
        identity = x
        x = x.to(self.device)
        identity = identity.to(self.device)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)


        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity


        return x