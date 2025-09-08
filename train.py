import os
import random
import warnings

warnings.filterwarnings("ignore")
import yaml
import time, torch, sys
from tqdm import tqdm
import numpy as np
from data.dataset_e2e import get_dataloaders
import time
from sklearn.metrics import roc_curve, auc
from models.OddMatcher import *
from models.losses import *
from models.config import CfgNode
import datetime
import argparse
import lovely_tensors

lovely_tensors.monkey_patch()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_filename_datetime():
    now = datetime.datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S")  # Example format: YYYYMMDD_HHMMSS
    return filename


def build_network(args, config, steps):
    model = OddMatcher(config, steps)
    model = model.to(args.device)

    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"], strict=False)
        print("model checkpoint loaded correctly...")

    return model


def evaluate(test_dataset, model):
    model.eval()
    
    gt_labels_list = []
    pred_labels_list = []
    pred_score_list = []

    for batch in tqdm(test_dataset):

        with torch.no_grad():
            rgb_images = batch['rgb_images'].cuda()
            targets = batch['targets']
            cam_RT = batch['cam_RT']
            cam_K = batch['cam_K']
            ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()
            gt_labels, pred_labels, pred_logits = model.test(rgb_images, ref_cam_P, targets)

            gt_labels_list.append(gt_labels)
            pred_labels_list.append(pred_labels.squeeze())
            pred_score_list.append(pred_logits.sigmoid().detach())

    gt_labels_list = torch.concat(gt_labels_list)
    pred_labels_list = torch.concat(pred_labels_list)
    pred_score_list = torch.concat(pred_score_list)

    total_acc = ((gt_labels_list==pred_labels_list)).float().mean()
    fpr, tpr, _ = roc_curve(gt_labels_list.cpu(), pred_score_list.cpu())
    roc_auc = auc(fpr, tpr)

    model.train()

    return total_acc, roc_auc


def evaluate_anomaly(test_dataset, model, selected_anom):
    model.eval()
    
    gt_labels_list = []
    pred_labels_list = []
    pred_score_list = []
    anom_labels_list = []

    for batch in tqdm(test_dataset):

        with torch.no_grad():
            try: 
                rgb_images = batch['rgb_images'].cuda()
                targets = batch['targets']
                cam_RT = batch['cam_RT']
                cam_K = batch['cam_K']
                ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()

                anom_types = batch['anom_types']
                anom_current_only = [
                    [1 if item == selected_anom else 0 for item in sublist]
                    for sublist in anom_types
                ]

                anom_current_only = [torch.tensor(seq, dtype=torch.long).to('cuda', non_blocking=True) for seq in anom_current_only]

                gt_labels, pred_labels, pred_logits, anom_labels = model.test_anomalies(rgb_images, ref_cam_P, targets, selected_anomaly_labels=anom_current_only)

                gt_labels_list.append(gt_labels)
                pred_labels_list.append(pred_labels.squeeze())
                pred_score_list.append(pred_logits.sigmoid().detach())
                anom_labels_list.append(anom_labels)
            except:
                print("Error in training loop...")

    ### Computing accuracies (along with precision, recall and roc)
    gt_labels_list = torch.concat(gt_labels_list)
    pred_labels_list = torch.concat(pred_labels_list)
    pred_score_list = torch.concat(pred_score_list)
    anom_labels_list = torch.concat(anom_labels_list)

    # I wanna compute only the accuracy regarding anomalies specified among the arguments
    anomaly_specific_acc = ((gt_labels_list==pred_labels_list)[anom_labels_list == 1]).float().mean()

    model.train()

    return anomaly_specific_acc


def train(train_dataset, test_dataset, model, device, dataset_name='toys', epochs=50):
    i = 0

    if dataset_name == 'toys':
        [seen_test_dataset, unseen_test_dataset] = test_dataset

    for epoch in range(epochs):

        print ('#Epoch - ' + str(epoch))
        start_time = time.time()

        for batch in train_dataset:
            i = i + 1
            rgb_images = batch['rgb_images'].to(device, non_blocking=True)
            targets = batch['targets']
            cam_RT = batch['cam_RT']
            cam_K = batch['cam_K']
            ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).to(device, non_blocking=True)

            loss_dict, log_dict = model(rgb_images, ref_cam_P, targets)
            model.scheduler.step()
            
            if i%100 == 0:
                print (f'[Epoch:{epoch}] [Step:{i}] [Time:{np.round(time.time() - start_time, 1)}s] logged info: { {i:np.round(j.item(),3) if isinstance(j, torch.Tensor) else np.round(j,3) for i,j in ({**loss_dict, **log_dict}).items()}}')
                start_time = time.time()


        if (epoch+1) % 10 == 0:
            if dataset_name == 'toys':
                acc_seen, roc_seen = evaluate(seen_test_dataset, model)
                acc_unseen, roc_unseen = evaluate(unseen_test_dataset, model)
                print(f'Acc SEEN#{epoch} - ACC_SEEN: {acc_seen}, ROC_SEEN: {roc_seen}')
                print(f'Acc SEEN#{epoch} - ACC_SEEN: {acc_unseen}, ROC_SEEN: {roc_unseen}')

            else:
                acc, roc = evaluate(test_dataset, model)
                print(f'Acc SEEN#{epoch} - ACC: {acc}, ROC: {roc}')
        
        if (epoch+1)%5 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": model.optim_alpha.state_dict(),
                    "epoch": epoch,
                },
                args.ckpt_path + f"/model_{str(epoch+1).zfill(6)}.pt"
            )


def main(args, config):
    #set_seed(args.seed)

    if args.config_path.split("/")[-1] == 'default.yaml':
        dataset_name = 'toys'
    else:
        dataset_name = 'parts'

    train_dataset, test_dataset = get_dataloaders(config, dataset_name=dataset_name)
    model = build_network(args, config, steps=len(train_dataset)*args.epochs)
    train(train_dataset, test_dataset, model, args.device, dataset_name=dataset_name, epochs=args.epochs)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/conf_toys.yaml')
    parser.add_argument('--name_run', type=str, default=f'run')
    #parser.add_argument('--seed', type=int, default=42)
    
    cur_time = get_filename_datetime()
    now = datetime.datetime.now()
    args = parser.parse_args()

    with open(args.config_path) as file:
        config = CfgNode(yaml.safe_load(file))

    args.ckpt_path = f'experiments/{args.name_run}_{cur_time}/checkpoints/'
    os.makedirs(args.ckpt_path, exist_ok=True)

    with open(f'experiments/command', 'w') as f:
        f.write(" ".join(sys.argv[:]))

    main(args, config)