import numpy as np
import utils.general as utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import distributions as dist


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet_VAE(nn.Module):
    ''' PointNet-based encoder network. Based on: https://github.com/autonomousvision/occupancy_networks

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        torch.nn.init.constant_(self.fc_mean.weight,0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):

        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean,c_std

class Decoder(nn.Module):
    '''  Based on: https://github.com/facebookresearch/DeepSDF
    '''
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        activation=None,
        latent_dropout=False,
    ):
        super().__init__()


        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3
            lin = nn.Linear(dims[l], out_dim)

            if (l in dropout):
                p = 1 - dropout_prob
            else:
                p = 1.0

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=2*np.sqrt(np.pi) / np.sqrt(p * dims[l]), std=0.000001)
                torch.nn.init.constant_(lin.bias, -1.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(p*out_dim))

            if weight_norm and l in self.norm_layers:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.use_activation = not activation == 'None'

        if self.use_activation:
            self.last_activation = utils.get_class(activation)()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1) /np.sqrt(2)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)/np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_activation:
            x = self.last_activation(x) + 1.0 * x

        return x

class SALNetwork(nn.Module):
    def __init__(self,conf,latent_size):
        super().__init__()
        self.decoder = Decoder(latent_size=latent_size,**conf.get_config('decoder'))
        if (latent_size > 0):
            self.encoder = SimplePointnet_VAE(hidden_dim=2*latent_size,c_dim=latent_size)
        else:
            self.encoder = None
        self.decode_mnfld_pnts = conf.get_bool('decode_mnfld_pnts')

    def forward(self, non_mnfld_pnts,mnfld_pnts):
        if not self.encoder is None:
            q_latent_mean,q_latent_std = self.encoder(mnfld_pnts)
            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = 1.0e-3*(q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))

            # Out of manfiold points
            non_mnfld_pnts = torch.cat([latent.unsqueeze(1).repeat(1, non_mnfld_pnts.shape[1], 1),
                                                    non_mnfld_pnts], dim=-1)
        else:
            latent_reg = None


        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1]))
        manifold_pnts_pred =  self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])) if (self.decode_mnfld_pnts) else None

        return {"manifold_pnts_pred":manifold_pnts_pred,
                "nonmanifold_pnts_pred":nonmanifold_pnts_pred,
                "latent_reg" : latent_reg}
