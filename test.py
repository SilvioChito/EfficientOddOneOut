from data.dataset_e2e import get_AnomalyTesting_dataloader, get_CountTesting_dataloader
from train import *
import numpy as np


def profile_inference(model, input_data, device='cuda'):
    model.to(device).eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_data = next(iter(input_data))

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # Timing inference
    with torch.no_grad():
        rgb_images = input_data['rgb_images'][0].unsqueeze(0).cuda()
        targets = input_data['targets']
        cam_RT = input_data['cam_RT'][0]
        cam_RT = np.expand_dims(cam_RT, axis=0)
        cam_K = input_data['cam_K'][0]
        cam_K = np.expand_dims(cam_K, axis=0)
        ref_cam_P = torch.Tensor(cam_K@cam_RT[:,:,:3]).cuda()

        torch.cuda.synchronize()

        starter.record()
        _, _, _ = model.test(rgb_images, ref_cam_P, targets)
        ender.record()

        torch.cuda.synchronize()
    
        elapsed_time_ms = starter.elapsed_time(ender)
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"[INFO] Inference time: {elapsed_time_ms:.2f} ms")
    print(f"[INFO] Max memory allocated: {max_memory_mb:.2f} MB")

    return elapsed_time_ms, max_memory_mb




if __name__ == "__main__":

    cur_time = get_filename_datetime()

    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--config_path', type=str, default='configs/conf_toys.yaml')
    parser.add_argument('--name_run', type=str, default=f'run_{cur_time}')

    args = parser.parse_args()
 
    with open(args.config_path) as file:
        config = CfgNode(yaml.safe_load(file))

    if args.config_path.split("/")[-1] == 'default.yaml':
        dataset_name = "toys"
    else:
        dataset_name = "parts"
    
    train_dataset, test_dataset = get_dataloaders(config, dataset_name=dataset_name)
    model = build_network(args, config, steps=len(train_dataset)*args.epochs)


    ### TESTING EVALUATION METRICS ON TOYS AND PARTS
    if dataset_name == "toys":
        [seen_test_dataset, unseen_test_dataset] = test_dataset

        total_acc_seen, roc_auc_seen = evaluate(seen_test_dataset, model)
        print('SEEN TEST SET')
        print(f'total_acc: {total_acc_seen}, roc_auc: {roc_auc_seen}')
        
        total_acc_unseen, roc_auc_unseen = evaluate(unseen_test_dataset, model)
        print('UNSEEN TEST SET')
        print(f'total_acc: {total_acc_unseen}, roc_auc: {roc_auc_unseen}')
    else:
        total_acc, roc_auc = evaluate(test_dataset, model)
        print(f'total_acc: {total_acc}, roc_auc: {roc_auc}')


    ### EVALUATING INFERENCE
    profile_inference(model, test_dataset, device=args.device)

        
    ### TESTING ANOMALY ACCURACIES FOR TOYS AND PARTS
        
    # Anomalies full names:
        # Anomaly types are derived from the paths of the single objects inside the scenes 
        # (Then I adapted them in order to be in a more readable format... based on the original paper

        # 'miss' => 'Missing Anomaly'
        # 'disc' => 'Material Mismatch Anomaly'
        # 'frac' => 'Fracture Anomaly'
        # 'bump' => 'Bump Anomaly'
        # 'rota' => 'Rotation Anomaly'
        # 'crack' => 'Crack Anomaly'
        # 'defr' => 'Deformation Anomaly'
        # 'tran' => 'Translation Anomaly'
        # 'diff' => 'Out of Category Distribution Anomaly'

    if dataset_name == "toys":
        toys_anomalies = ['disc', 'defr', 'crack', 'frac', 'rota', 'miss', 'bump', 'tran']

        print(f'ANOMALIES ACCURACIES TOYS:')
        for anom in toys_anomalies:
            anom_dataset = get_AnomalyTesting_dataloader(config, anom, dataset_name=dataset_name)
            anom_acc = evaluate_anomaly(anom_dataset, model, anom)
            print(f'{anom} acc: {anom_acc}')

    else:
        parts_anomalies = ['bump', 'diff', 'crack', 'frac']

        print(f'ANOMALIES ACCURACIES PARTS:')
        for anom in parts_anomalies:
            anom_dataset = get_AnomalyTesting_dataloader(config, anom, dataset_name=dataset_name)
            anom_acc = evaluate_anomaly(anom_dataset, model, anom)
            print(f'{anom} acc: {anom_acc}')
    


    ### TESTING AUC FOR TOYS AND PARTS BASED ON THE NUMBER OF OBJECTS INSIDE THE SCENE
    ### ONLY ON PARTS (Since it has more objects inside the scenes)###
    if dataset_name == "parts":
        obj_count = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
        for counts in obj_count:
            count_dataset = get_CountTesting_dataloader(config, count_obj=counts, dataset_name=dataset_name)
            _, roc_auc = evaluate(count_dataset, model)
            print(f'roc_auc for counts {counts}: {roc_auc}')

