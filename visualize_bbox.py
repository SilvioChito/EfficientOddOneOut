import copy
from train import *
from utils.drawbox import *
import numpy as np
import tqdm
import argparse

def denormalize_batch(batch_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Args:
        batch_tensor: Tensor of shape (B, C, H, W) - normalized RGB images
        mean: tuple of 3 floats for each channel
        std: tuple of 3 floats for each channel
    Returns:
        Tensor of shape (B, C, H, W) - denormalized RGB images
    """
    mean = torch.tensor(mean, device=batch_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=batch_tensor.device).view(1, 3, 1, 1)
    return batch_tensor * std + mean


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='moad-rendering_nvs_dinov2_full_dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/conf_toys.yaml')
    parser.add_argument('--name_run', type=str, default=f'run')
    parser.add_argument('--out_path', type=str, default=f'./')

    #parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    with open(args.config_path) as file:
        config = CfgNode(yaml.safe_load(file))

    if args.config_path.split("/")[-1] == 'default.yaml':
        dataset_name = 'toys'
        train_dataset,  [seen_test_loader, unseen_test_loader] = get_dataloaders(config, dataset_name=dataset_name)
    else: 
        dataset_name = 'parts'
        train_dataset, test_dataset = get_dataloaders(config, dataset_name=dataset_name)

    model = build_network(args, config, steps=len(train_dataset) * args.epochs)
    output_path = args.out_path
    os.makedirs(args.out_path, exist_ok=True)

    count = 0
    for batch in tqdm.tqdm(test_dataset):

        with torch.no_grad():
            rgb_images = batch['rgb_images'].cuda()
            rgb_images = denormalize_batch(rgb_images)
            target_mask = batch['batch_silhouettes'].cuda()
            targets = batch['targets']
            cam_RT = batch['cam_RT']
            cam_K = batch['cam_K']
            scene_path = batch['scene_path']
            ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()
            bs, num_view, _, size, size = rgb_images.shape
            gt_labels, pred_labels, pred_logits = model.test(rgb_images, ref_cam_P, targets)

            batched_pls = pred_labels.squeeze().split([i['labels'].shape[0] for i in targets])
            predict_dict = copy.deepcopy(targets)

            for batch_idx,_ in enumerate(predict_dict):
                predict_dict[batch_idx]['labels']=batched_pls[batch_idx]
                all_views = []
                input_views = []

                for view_idx in np.arange(len(rgb_images[batch_idx]))[::2]:
                    input_rgb_vis = rgb_images[batch_idx][view_idx].permute(1,2,0).detach().cpu().numpy()*255

                    tgt_vis = draw_boxes(rgb_images[batch_idx][view_idx].unsqueeze(0), \
                        ref_cam_P[batch_idx][view_idx].unsqueeze(0), \
                            targets[batch_idx]['boxes'], targets[batch_idx]['labels'])
                    
                    pred_vis = draw_boxes(rgb_images[batch_idx][view_idx].unsqueeze(0), \
                        ref_cam_P[batch_idx][view_idx].unsqueeze(0), \
                            predict_dict[batch_idx]['boxes'], predict_dict[batch_idx]['labels'])
                    
                    merged_vis = np.concatenate([tgt_vis, pred_vis], 0)

                    all_views.append(merged_vis)
                    input_views.append(input_rgb_vis)
                
                res_bbox_vis = np.concatenate(all_views,1)
                input_views_vis = np.concatenate(input_views,1)
                name = os.path.basename(scene_path[batch_idx])
                cv2.imwrite(output_path+'/'+name+'.png', res_bbox_vis)
                print(output_path+'/'+name+'.png')


objs = []
ans = []
for i in range(10000):
    num_obj = random.randint(3, 5+1)
    num_anomaly = random.randint(1, num_obj//2)
    objs.append(num_obj)
    ans.append(num_anomaly)