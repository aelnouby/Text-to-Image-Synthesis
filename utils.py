import numpy as np
from torch import nn
import torch
import pdb

class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2,  3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


class minibatch_discriminator(nn.Module):
    def __init__(self, num_channels, B_dim, C_dim):
        super(minibatch_discriminator, self).__init__()
        self.B_dim = B_dim
        self.C_dim =C_dim
        self.num_channels = num_channels
        T_init = torch.randn(num_channels * 4 * 4, B_dim * C_dim) * 0.1
        self.T_tensor = nn.Parameter(T_init, requires_grad=True)

    def forward(self, inp):
        inp_flat = inp.view(-1, self.num_channels * 4 * 4)
        M = inp_flat.mm(self.T_tensor)
        M = M.view(-1, self.B_dim, self.C_dim)

        op1 = M.unsqueeze(3)
        op2 = M.permute(1, 2, 0).unsqueeze(0)

        output = torch.sum(torch.abs(op1 - op2), 2)
        output = torch.sum(torch.exp(-output), 2)
        output = output.view(M.size(0), -1, 4, 4)

        output = torch.cat((inp, output), 1)

        return output


class Utils(object):
    def smooth_label(tensor, offset):
        indices = np.random.choice(range(len(tensor)), int(len(tensor) / 2.0))
        prob = np.random.rand(len(indices))

        tensor[indices] = tensor[indices] + (offset * prob)

        return tensor
