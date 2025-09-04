
import torch
from torch import nn
import einops as ei


class NormalDeviationLoss(nn.Module):
    def __init__(self):
        super(NormalDeviationLoss, self).__init__()
        self.loss = torch.nn.functional.mse_loss

    def forward(self, x, padded_gt, ref):
        """
        x: [B, N, C] - Embeddings
        padded_gt: [B, N] - Padded mask
        """
        total_loss = 0
        normality_mask = padded_gt == 0
        masked_x = torch.where(normality_mask[..., None], x, 0)
        normal_proto = ei.reduce(masked_x, 'b n c -> b c', reduction='sum') / normality_mask.sum(-1, keepdims=True)
        total_loss = self.loss(normal_proto, ref)

        return total_loss