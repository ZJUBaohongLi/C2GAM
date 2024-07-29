import torch
import torch.nn as nn
import torch.utils
import torch.distributions as D

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        dim_in = config.get('encoder_dim_in')
        dim_latent = config.get('encoder_dim_latent')
        dim_out = config.get('encoder_dim_out')
        layer_num = config.get('encoder_layer_num')
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in - (dim_in - dim_out) * i // layer_num,
                          dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.BatchNorm1d(dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.ReLU(),
            ) for i in range(layer_num)
        ])
        self.mean_layer = nn.Linear(dim_out, dim_latent)
        self.sigma_layer = nn.Linear(dim_out, dim_latent)
        self.noise_distribution = D.Normal(0, 1)
        self.noise_distribution.loc = self.noise_distribution.loc.to(device)
        self.noise_distribution.scale = self.noise_distribution.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x_rep = x.to(torch.float32)
        for layer in self.mlp:
            x_rep = layer(x_rep)
        mean = self.mean_layer(x_rep)
        sigma = torch.exp(self.sigma_layer(x_rep))
        z = mean + sigma * self.noise_distribution.sample(mean.shape)
        self.kl = (sigma ** 2 + mean ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        dim_in = config.get('encoder_dim_in')
        dim_latent = config.get('encoder_dim_latent')
        dim_out = config.get('encoder_dim_out')
        layer_num = config.get('encoder_layer_num')
        self.rep_layer = nn.Sequential(
            nn.Linear(dim_latent, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(),
        )
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in - (dim_in - dim_out) * (i + 1) // layer_num,
                          dim_in - (dim_in - dim_out) * i // layer_num),
                nn.BatchNorm1d(dim_in - (dim_in - dim_out) * i // layer_num),
                nn.ReLU(),
            ) for i in range(layer_num - 1, 0, -1)
        ])
        self.output_layer = nn.Linear(dim_in - (dim_in - dim_out) // layer_num, dim_in)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, z, bin_feats):
        z_rep = z.to(torch.float32)
        z_rep = self.rep_layer(z_rep)
        for layer in self.mlp:
            z_rep = layer(z_rep)
        x = self.output_layer(z_rep)
        x[:, bin_feats] = self.sigmoid_layer(x[:, bin_feats])
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, config):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, bin_feats):
        z = self.encoder(x)
        ret = self.decoder(z, bin_feats)
        return ret
