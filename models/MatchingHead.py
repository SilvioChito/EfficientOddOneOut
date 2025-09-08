import torch
import torch.nn as nn
from einops import repeat 
import einops as ei
from models.losses import NormalDeviationLoss
from torch.nn.utils.rnn import pad_sequence
DEBUG_PERM = True


class ResidualHead(nn.Module):
    def __init__(self, in_dim):
        super(ResidualHead, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Sequential(nn.Linear(in_dim, in_dim),
                                    nn.LayerNorm(in_dim),
                                    nn.ReLU())
    def forward(self, x, ref):
        """
        Returns residual anomalies by comparing with the normality prototype

        Args:
            x (embeddings representing objs in the scene): b x n x d
            ref (normality prototype embedding): b x d

        Returns:
            residual_anomalies: b x n x d
        """

        residual_anomalies = ref[:, None, :] - self.conv(x)
        return residual_anomalies



class  MatchingHead(nn.Module):
    def __init__(self, config):
        super(MatchingHead, self).__init__()

        input_dim = config.model.oddmatcher.input_channel_dim
        squeezed_input_dim = config.model.oddmatcher.squeezed_dim
        n_layer_attn = config.model.oddmatcher.n_layer_attn
        n_head = config.model.oddmatcher.n_head
        
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(squeezed_input_dim, nhead=n_head, dim_feedforward=squeezed_input_dim*2, batch_first=True)
            for _ in range(n_layer_attn)
        ])
    
        self.centroid_head = nn.TransformerEncoderLayer(squeezed_input_dim, nhead=n_head, dim_feedforward=squeezed_input_dim*2, batch_first=True)
        self.centroid_normality = nn.Parameter(torch.randn(squeezed_input_dim), requires_grad=True)
        self.proj =  nn.Linear(input_dim, squeezed_input_dim)
        self.cls_head = nn.Linear(squeezed_input_dim, 1)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.residual_head = ResidualHead(squeezed_input_dim)
        self.deviation_loss = NormalDeviationLoss()

    
    @torch.autocast(device_type='cuda')
    def forward(self, input, anom_labels=None):
        batch_inp, batch_gt = zip(*input)
        
        batch_inp = [ei.reduce(inp, 'n c z y x -> n c', reduction='mean') for inp in batch_inp]
        batch_inp = torch.nested.as_nested_tensor(batch_inp)

        batch_inp = self.proj(batch_inp)
        batch_inp = batch_inp.to_padded_tensor(padding=-100)

        if anom_labels is not None:
            anom_labels = pad_sequence(anom_labels, batch_first=True, padding_value=-100)

        B, N, D = batch_inp.shape
        padded_gt = torch.nested.as_nested_tensor([torch.tensor(b) for b in batch_gt], device=batch_inp.device)
        padded_gt = padded_gt.to_padded_tensor(padding=-100)

        if DEBUG_PERM and self.training:
            perm = torch.randperm(N, device=batch_inp.device)
            batch_inp = batch_inp[:, perm]
            padded_gt = padded_gt[:, perm]


        ### COMPUTING NORMALITY CENTROID FOR EACH SCENE
        feat_norm_token, _ = ei.pack([repeat(self.centroid_normality, 'C -> B 1 C', B=B), batch_inp], 'B * C')
        src_key_padding_mask = (feat_norm_token[:, :, 0] == -100).cuda()  # [B, M] mask employed for padding masking
        feat_list = [feat_norm_token]
        feat_list = self.centroid_head(feat_norm_token, src_key_padding_mask=src_key_padding_mask)

        norm_token = feat_list[:, 0, :]
        seq_feat = feat_list[:, 1:, :]
        normality_loss = self.deviation_loss(seq_feat, padded_gt, norm_token)


        ### MATCH CONTEXT HEAD
        mask = (batch_inp == -100)[:, :, 0]

        feat_list = [batch_inp]
        for layer in self.attn_layers:
            feat_list.append(layer(feat_list[-1], src_key_padding_mask=mask))
        
        feat_list, _ = ei.pack(feat_list, '* B N C')        
        final_seq_feat = feat_list[-1]

        final_seq_feat = self.residual_head(final_seq_feat, norm_token)  # Residual features already computed because we have already the final seq feats and normal token embeddings

        ### COMPUTING CONTRASTIVE & CLASSIFICATION LOSS OVER THE FINAL EMBEDDINGS        
        logits = self.cls_head(final_seq_feat)
        loss = self.cls_loss(logits[..., 0], padded_gt.to(logits.dtype))[padded_gt!=-100].mean()

        pred_labels = (logits.sigmoid()>0.5)[padded_gt!=-100].float()
        gt_labels = padded_gt[padded_gt!=-100].float()
        pred_logits = logits[padded_gt!=-100].float()
        batch_acc = (pred_labels.squeeze()==gt_labels).float().mean()

        if anom_labels is not None:
            anom_labels = anom_labels[padded_gt!=-100].float()

        if anom_labels is None:
            return loss, batch_acc, normality_loss, [pred_logits, pred_labels, gt_labels]
        else:
            return loss, batch_acc, normality_loss, [pred_logits, pred_labels, gt_labels, anom_labels]
    