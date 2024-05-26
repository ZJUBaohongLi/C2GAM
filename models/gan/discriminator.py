import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        dim_in = config.get('discriminator_dim_in')
        dim_latent = config.get('discriminator_dim_latent')
        dim_out = config.get('discriminator_dim_out')
        layer_num = config.get('discriminator_layer_num')
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in - (dim_in - dim_out) * i // layer_num,
                          dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.BatchNorm1d(dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.ReLU(),
            ) for i in range(layer_num)
        ])
        self.output_label_layer = nn.Sigmoid()
        self.output_layer = nn.Linear(dim_out, dim_latent)

    def forward(self, x):
        x_rep = x.to(torch.float32)
        for layer in self.mlp:
            x_rep = layer(x_rep)
        label = self.output_label_layer(self.output_layer(x_rep))
        return label

