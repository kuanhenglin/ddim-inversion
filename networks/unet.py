from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

import utilities.network as nutils


class Upsample(nn.Module):

    def __init__(self, in_channels, do_conv=True):
        super().__init__()
        self.do_conv = do_conv
        if do_conv:  # Use convolution for downsampling instead of linear interpolation
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.do_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels, do_conv=True):
        super().__init__()
        self.do_conv = do_conv
        if do_conv:  # Use convolution for downsampling instead of linear interpolation
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.do_conv:
            padding = (0, 1, 0, 1)  # Asymmetric padding (only on one side of each axis)
            x = F.pad(x, padding, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, do_conv_skip=False,
                 dropout=0.0, time_embed_channels=512, num_groups=32):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        # Convolution layer 1
        self.norm_1 = nutils.group_norm(in_channels, num_groups=num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if time_embed_channels is not None:
            self.time_embed_proj = nn.Linear(time_embed_channels, out_channels)
        else:
            self.time_embed_proj = None
        # Convolution layer 2
        self.norm_2 = nutils.group_norm(out_channels, num_groups=num_groups)
        self.dropout = nn.Dropout(p=dropout)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # ResNet skip layer (with 3x3 kernel option available with do_conv_skip)
        if in_channels != out_channels:
            kernel_size = 3 if do_conv_skip else 1  # Skip with larger kernel
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                       stride=1, padding=kernel_size // 2)
        else:  # Identity skip (do nothing)
            self.conv_skip = nn.Identity()

    def forward(self, x, t_embed=None):
        h = x  # Order for each layer: norm -> activation -> conv
        h = self.norm_1(h)
        h = nutils.swish(h)
        h = self.conv_1(h)

        if self.time_embed_proj is not None:
            if t_embed is None:
                raise ValueError("time_embed cannot be None when the block is initialized with"
                                 "time embedding projection.")
            t_embed = nutils.swish(t_embed)
            h = h + self.time_embed_proj(t_embed)[:, :, None, None]  # Apply to each channel

        h = self.norm_2(h)
        h = nutils.swish(h)
        h = self.dropout(h)  # Apply dropout on second convolution layer
        h = self.conv_2(h)

        x = self.conv_skip(x)
        h = x + h
        return h


class AttentionBlock(nn.Module):

    def __init__(self, in_channels, num_groups=32):
        super().__init__()

        self.norm = nutils.group_norm(in_channels, num_groups=num_groups)
        self.Q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.K = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.V = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm(h)
        q = self.Q(h)
        k = self.K(h)
        v = self.V(h)

        # Compute attention
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).moveaxis(-1, -2)  # B C H W -> B (H W) C
        k = k.reshape(B, C, H * W)  # B C H W -> B C (H W)
        w = q @ k  # Batched scores, B (H W) (H W), w[b, i, j] = sum_c q[b, i, c] k[b, c, j]
        w.mul_(pow(C, -0.5))  # Normalize scores
        w = F.softmax(w, dim=-1)  # Apply softmax in preparation for summing

        # Apply attention to values
        v = v.reshape(B, C, H * W)  # B C H W -> B C (H W)
        w = w.moveaxis(-1, -2)  # B (H W of q) (H W of k) -> B (H W of k) (H W of q)
        h = v @ w  # Batched attention, B C (H W) (H W of q), sum_i v[b, c, i] w[b, i, j]
        h = h.reshape(B, C, H, W)  # ^ Taking linear combination of values weighted by cores

        h = self.out_proj(h)
        h = x + h  # Residual skip connection
        return h


class UNet(nn.Module):

    def __init__(self, in_shape, hidden_channels, num_blocks, channel_mults, attention_sizes,
                 time_embed_channels, dropout=0.0, num_groups=16, do_conv_sample=True,
                 out_conv_zero=True):
        super().__init__()

        assert in_shape[1] == in_shape[2], f"Input shape must be square."

        self.in_shape = in_shape
        self.hidden_channels = hidden_channels
        self.num_sizes = len(channel_mults)
        self.num_blocks = num_blocks

        # Time embedding

        if time_embed_channels is not None:
            self.time_embed = nn.Module()
            self.time_embed.fn = partial(nutils.time_embed, embed_dim=hidden_channels)
            self.time_embed.dense = nn.ModuleList([
                nn.Linear(hidden_channels, time_embed_channels),
                nn.Linear(time_embed_channels, time_embed_channels)])
        else:
            self.time_embed = None

        # Downsampling layers

        self.in_conv = nn.Conv2d(in_shape[0], hidden_channels, kernel_size=3, stride=1, padding=1)

        current_size = in_shape[1]
        in_channel_mults = [1] + channel_mults
        self.down_layers = nn.ModuleList()
        in_channels_block = None
        for i in range(len(channel_mults)):
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            in_channels_block = round(hidden_channels * in_channel_mults[i])
            out_channels_block = round(hidden_channels * channel_mults[i])
            # Add num_blocks Resnet blocks (with Attention blocks)
            for _ in range(num_blocks):
                blocks.append(ResnetBlock(in_channels_block, out_channels_block,
                                          time_embed_channels=time_embed_channels,
                                          dropout=dropout, num_groups=num_groups))
                in_channels_block = out_channels_block
                if current_size in attention_sizes:
                    attentions.append(AttentionBlock(in_channels_block, num_groups=num_groups))
            # Create down layer as nn.Module
            down_layer = nn.Module()
            down_layer.blocks = blocks
            down_layer.attentions = attentions
            if i != len(channel_mults) - 1:  # Downsample unless at last layer
                down_layer.downsample = Downsample(in_channels_block, do_conv=do_conv_sample)
                current_size = current_size // 2
            self.down_layers.append(down_layer)

        # Middle layers

        self.mid_layers = nn.ModuleList()
        self.mid_layers.block_1 = ResnetBlock(in_channels_block, in_channels_block,
                                              time_embed_channels=time_embed_channels,
                                              dropout=dropout, num_groups=num_groups)
        self.mid_layers.attention = AttentionBlock(in_channels_block, num_groups=num_groups)
        self.mid_layers.block_2 = ResnetBlock(in_channels_block, in_channels_block,
                                              time_embed_channels=time_embed_channels,
                                              dropout=dropout, num_groups=num_groups)

        # Upsampling layers

        self.up_layers = nn.ModuleList()
        for i in reversed(range(len(channel_mults))):
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            out_channels_block = round(hidden_channels * channel_mults[i])
            in_channels_skip = round(hidden_channels * channel_mults[i])
            for j in range(num_blocks + 1):
                if j == num_blocks:
                    in_channels_skip = hidden_channels * in_channel_mults[i]
                blocks.append(ResnetBlock(in_channels_block + in_channels_skip, out_channels_block,
                                          time_embed_channels=time_embed_channels,
                                          dropout=dropout, num_groups=num_groups))
                in_channels_block = out_channels_block
                if current_size in attention_sizes:
                    attentions.append(AttentionBlock(in_channels_block, num_groups=num_groups))
            # Create up layer as nn.Module
            up_layer = nn.Module()
            up_layer.blocks = blocks
            up_layer.attentions = attentions
            if i != 0:
                up_layer.upsample = Upsample(in_channels_block, do_conv=do_conv_sample)
                current_size *= 2
            self.up_layers.insert(0, up_layer)

        # End layers

        self.out_norm = nutils.group_norm(in_channels_block, num_groups=num_groups)
        self.out_conv = nn.Conv2d(in_channels_block, in_shape[0],
                                  kernel_size=3, stride=1, padding=1)
        if out_conv_zero:
            self.out_conv.weight.data.fill_(0.0)

    def forward(self, x, t=None):
        assert list(x.shape[-3:]) == self.in_shape, \
               f"Shape of x {tuple(x.shape)} does not match network definition."

        # Time embedding
        if self.time_embed is not None:
            if t is None:
                raise ValueError("t cannot be None when the U-Net is initialized with"
                                 "time embedding projection.")
            t_embed = self.time_embed.fn(t)
            t_embed = self.time_embed.dense[0](t_embed)
            t_embed = nutils.swish(t_embed)
            t_embed = self.time_embed.dense[1](t_embed)
        else:
            t_embed = None

        # Downsampling

        h_skip = [self.in_conv(x)]
        for i in range(self.num_sizes):
            for j in range(self.num_blocks):
                h = self.down_layers[i].blocks[j](h_skip[-1], t_embed)
                if len(self.down_layers[i].attentions) > 0:  # Apply attention heads
                    h = self.down_layers[i].attentions[j](h)
                h_skip.append(h)
            if i != self.num_sizes - 1:
                h = self.down_layers[i].downsample(h_skip[-1])
                h_skip.append(h)

        # Middle

        h = h_skip[-1]
        h = self.mid_layers.block_1(h, t_embed)
        h = self.mid_layers.attention(h)
        h = self.mid_layers.block_2(h, t_embed)

        # Upsampling

        for i in reversed(range(self.num_sizes)):
            for j in range(self.num_blocks + 1):
                h = torch.cat([h, h_skip.pop()], dim=-3)  # Concatenate with skip at channel
                h = self.up_layers[i].blocks[j](h, t_embed)
                if len(self.up_layers[i].attentions) > 0:  # Apply attention heads
                    h = self.up_layers[i].attentions[j](h)
            if i != 0:
                h = self.up_layers[i].upsample(h)

        # End

        h = self.out_norm(h)
        h = nutils.swish(h)
        h = self.out_conv(h)
        return h
