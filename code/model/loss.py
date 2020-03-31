from torch import nn
import torch
import utils.general as utils
import numpy as np
#
class GenLoss(nn.Module):
    def __init__(self,manifold_pnts_weight):
        super().__init__()
        self.manifold_pnts_weight = manifold_pnts_weight

class SALLoss(GenLoss):
    def __init__(self, manifold_pnts_weight,unsigned):
        super().__init__(manifold_pnts_weight)
        self.unsigned = unsigned

    def forward(self, manifold_pnts_pred,
                nonmanifold_pnts_pred,
                nonmanifold_gt,
                weight=None,
                latent_reg=None):

        if self.unsigned:
            recon_term = torch.abs(nonmanifold_pnts_pred.squeeze().abs() - nonmanifold_gt)
        else:
            recon_term = torch.abs(nonmanifold_pnts_pred.squeeze() - nonmanifold_gt)

        loss = recon_term.mean()
        if latent_reg is not None:
            loss = loss + latent_reg.mean()
            reg_term = latent_reg.mean().detach()
        else:
            reg_term = torch.tensor([0.0])
        return {"loss": loss, 'recon_term': recon_term.mean(), 'reg_term': reg_term.mean()}


class SALReconLoss(GenLoss):
    def __init__(self, manifold_pnts_weight,unsigned):
        super().__init__(manifold_pnts_weight)
        self.unsigned = unsigned

    def forward(self, manifold_pnts_pred,
                nonmanifold_pnts_pred,
                nonmanifold_gt,
                delta = None,
                weight=None,
                latent_reg=None):



        recon_term = torch.abs(nonmanifold_pnts_pred.squeeze().abs() - 1).mean() + self.manifold_pnts_weight*manifold_pnts_pred.abs().mean()

        loss = recon_term

        return {"loss": loss, 'recon_term': recon_term.mean(), 'reg_term': torch.tensor([0.0])}
