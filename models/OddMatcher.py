import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MultiViewEncoder import *
from models.MatchingHead import *

def rotate_feat_map(target_volume, R, t):

    B, C, D, H, W = target_volume.shape
    transformed_coordinates = F.affine_grid(torch.concat([R,t[:,:,None]], -1), (B, C, D, H, W))
    transformed_volume = torch.nn.functional.grid_sample(target_volume.cuda(), transformed_coordinates.cuda())
    return transformed_volume

def crop_objs(volume, targets, resize_dim = (8,8,8), voxel_size = 0.04, pad_len = 1):
    B, C, d_vox, h_vox, w_vox = volume.shape

    batch_list = []

    for b_id in range(B):
        
        objs_crops = []
        objs_labs = []
        boxes = targets[b_id]['boxes']
        bboxs = boxes/voxel_size
        bboxs[:,:3]+=torch.Tensor([[h_vox//2,w_vox//2,d_vox//2]])
        bboxs = bboxs.round().int()
        labels = targets[b_id]['labels']
        
        for box_i, lab_i in zip(bboxs, labels):
            x,y,z,_,_,_,pad = *box_i, pad_len
            w,h,l = bboxs[:,3:].max(0).values
            half_l, half_h, half_w = l//2+pad, h//2+pad, w//2+pad

            z_start = z - half_l
            z_end = z + half_l
            y_start = y - half_h
            y_end = y + half_h
            x_start = x - half_w
            x_end = x + half_w
            
            vol_crop = volume[b_id, :, z_start:z_end, y_start:y_end, x_start:x_end].clone()

            try:
                resized_crop = F.interpolate(vol_crop[None], size = resize_dim, mode = 'trilinear')
                objs_crops.append(resized_crop)
                objs_labs.append(lab_i)
                
            except:
                print("Errors while cropping single objects volumes...")
        
        batch_list.append([torch.concat(objs_crops), objs_labs])

    return batch_list



class OddMatcher(nn.Module):
    def __init__(self, config, steps):

        super(OddMatcher, self).__init__()
        self.multiview_enc = MultiViewEncoder(config)
        self.matching_head = MatchingHead(config)

        params = [{"params":self.multiview_enc.parameters(), "lr":float(config.train.lr_encoder)},
                  {"params":self.matching_head.parameters(), "lr":float(config.train.lr_oddmatcher)}]
        
        self.optim_alpha = torch.optim.AdamW(params, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_alpha, T_max=steps, eta_min=1e-8)
        self.scaler = torch.cuda.amp.GradScaler()


    def forward(self, rgb_images, ref_cam_P, targets):
        loss_dict, log_dict = {}, {}
        num_view = 5

        with torch.cuda.amp.autocast():
            rgb_images = rgb_images.half()
            ref_cam_P = ref_cam_P.half()
            
            scene_volume_feats = self.multiview_enc(rgb_images[:,:num_view], ref_cam_P[:,:num_view])
            obj_volumes = crop_objs(scene_volume_feats, targets, pad_len=1)
            cls_loss, batch_acc, normality_loss, _ = self.matching_head(obj_volumes)

            loss_dict["normality_loss"] = normality_loss
            loss_dict["cls_loss"] = cls_loss
            log_dict["batch_acc"] = batch_acc

            self.loss_total = sum(loss_dict.values())
            
        self.optim_alpha.zero_grad()
        self.scaler.scale(self.loss_total).backward()
        self.scaler.step(self.optim_alpha)
        self.scaler.update()

        return loss_dict, log_dict


    def test(self, rgb_images, ref_cam_P, targets):

        with torch.no_grad():

            scene_volume_feats = self.multiview_enc(rgb_images[:,:], ref_cam_P[:,:])
            obj_volumes = crop_objs(scene_volume_feats, targets, pad_len=1)
            _, _, _, preds = self.matching_head(obj_volumes)
            pred_logits, pred_labels, gt_labels = preds

        return gt_labels, pred_labels, pred_logits
      

    def test_anomalies(self, rgb_images, ref_cam_P, targets, selected_anomaly_labels=None):

        with torch.no_grad():

            scene_volume_feats = self.multiview_enc(rgb_images[:,:], ref_cam_P[:,:])
            obj_volumes = crop_objs(scene_volume_feats, targets, pad_len=1)
            _, _, _, preds = self.matching_head(obj_volumes, anom_labels=selected_anomaly_labels)
            pred_logits, pred_labels, gt_labels, anom_labels = preds

        return gt_labels, pred_labels, pred_logits, anom_labels
        