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


class Utils(object):
    def smooth_label(tensor, offset):
        indices = np.random.choice(range(len(tensor)), int(len(tensor) / 2.0))
        prob = np.random.rand(len(indices))

        tensor[indices] = tensor[indices] + (offset * prob)

        return tensor
