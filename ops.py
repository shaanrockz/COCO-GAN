"""
Most of the codes are from:
1. https://github.com/carpedm20/DCGAN-tensorflow
2. https://github.com/minhnhat93/tf-SNDCGAN
"""
import numpy as np 

import torch
import torch.nn as nn
import torch.functional as F

def _l2normalize(v, eps=1e-12):
  return v / (torch.sum(v ** 2) ** 0.5 + eps)


def lrelu(x, leak=0.2, name="lrelu"):
    return torch.max(x, leak*x)

def upscale(x, scale):
    return nn.Upsample(scale_factor=scale, mode='nearest')(x)


def pad(x, p):
    c = torch.tensor([[0, 0], [p, p,], [p, p], [0, 0]])
    return F.pad(x, c, mode='reflect')

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def add_coords(input_tensor, x_dim=64, y_dim=64, with_r=False):
    """
    For CoordConv.

    Add coords to a tensor
    input_tensor: (batch, x_dim, y_dim, c)
    """
    batch_size_tensor = input_tensor.size(0)
    
    xx_ones = torch.ones((batch_size_tensor, x_dim), dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)
    xx_range = tile(torch.range(0,x_dim-1).unsqueeze(0), 1, batch_size_tensor)
    xx_range = xx_range.unsqueeze(1)
    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)
    
    yy_ones = torch.ones((batch_size_tensor, y_dim),
        dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)
    yy_range = tile(torch.range(0,y_dim-1).unsqueeze(0), 1, batch_size_tensor)
    yy_range = yy_range.unsqueeze(-1)
    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)
    
    xx_channel = xx_channel.float() / (x_dim - 1)
    yy_channel = yy_channel.float() / (y_dim - 1)
    xx_channel = xx_channel*2 - 1
    yy_channel = yy_channel*2 - 1
    
    ret = torch.cat((input_tensor,
        xx_channel,
        yy_channel), -1)
        
    if with_r:
        rr = torch.sqrt( torch.pow(xx_channel-0.5, 2)
                + torch.pow(yy_channel-0.5, 2))
        ret = torch.cat((ret, rr), -1)
    return ret

