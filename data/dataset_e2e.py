import os
import json, glob
import numpy as np
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def apply_random_rotation_in_space(P, XYZ, r):
    r_inv = r.inverse()
    rot_P = [P_i@r for P_i in P]
    rot_XYZ = [(r_inv@torch.cat([XYZ_i,torch.ones([1])]))[:3] for XYZ_i in XYZ]
    return torch.stack(rot_P), torch.stack(rot_XYZ)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_transform(size=256, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    
    transform_list.append(transforms.Resize((size, size), interpolation=method))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    return transforms.Compose(transform_list)



class OddOneOutDataset(Dataset):
    def __init__(self, config, mode='train', dataset_name='toys', test_anomaly_type=None, count_obj=None):

        path =  config.dataset.data_path
        self.num_views = config.dataset.num_views
        self.dimension = config.dataset.dimension
        self.max_objs = config.dataset.max_objs
        self.random_scene_rotation_augmentation = config.dataset.random_scene_rotation_augmentation
        self.dataset_name = dataset_name

        self.all_scene_paths = [i for i in sorted(glob.glob(path+'/*')) if os.path.isfile(i+'/scene3d.metadata.json')]

        self.seen_categories_toys = ["dinosaur", "fish", "frog", "monkey", "light", "lizard", "orange", "boat", "dog", "lion", "pig", "cookie", "panda", "chicken",
                                "orange", "ice", "horse", "car", "airplane", "cake", "shark", "donut", "hat", "cow", "apple", "bowl", "hamburger", "octopus",
                                "giraffe", "chess", "bread", "butterfly", "cupcake", "bunny", "elephant", "fox", "deer", "bus", "bottle", "hammer", "mug", "key"]
        
        self.unseen_categories_toys = ["plate", "robot", "glass", "sheep", "shoe", "train", "banana", "cup", "penguin"]

        if test_anomaly_type is not None or count_obj is not None:

            if dataset_name == 'toys':
                unseen_scenes = [path for path in self.all_scene_paths if path.split("/")[-1].split("_")[0] in self.unseen_categories_toys]
            else:
                unseen_scenes = self.all_scene_paths[int(len(self.all_scene_paths)*0.80):]

            if test_anomaly_type is not None:
                self.scene_paths = [path for path in unseen_scenes if contain_anomaly(path, test_anomaly_type, dataset_name)]
            else:
                self.scene_paths = [path for path in unseen_scenes if contain_n_obects(path, count_obj)]

        else:
            if 'parts' in dataset_name:
                if mode == 'train':
                    self.scene_paths = self.all_scene_paths[:int(len(self.all_scene_paths)*0.80)]
                    print(f'len train dataset: {len(self.scene_paths)} scenes')
                else:
                    self.scene_paths = self.all_scene_paths[int(len(self.all_scene_paths)*0.80):]
                    print(f'len test dataset: {len(self.scene_paths)} scenes')


            elif 'toys' in dataset_name:
                train_scenes = [path for path in self.all_scene_paths if path.split("/")[-1].split("_")[0] in self.seen_categories_toys]
                unseen_scenes = [path for path in self.all_scene_paths if path.split("/")[-1].split("_")[0] in self.unseen_categories_toys]
                
                dic_train_cat = {} # dictionary with categories for train
                dic_seen_test_cat = {} # dictionary with categories for SEEN test

                for path in train_scenes:
                    category = path.split("/")[-1].split("_")[0]
                    if category in dic_train_cat:
                        dic_train_cat[category].append(path)
                    else:
                        dic_train_cat[category] = [path]

                cut_off = 0.15 # percentage w.r.t. the number of samples in train set, in order to obtain the proportion mentioned in original OddOneOut approach
                for cat in dic_train_cat.keys():
                    dic_seen_test_cat[cat] = dic_train_cat[cat][:int(len(dic_train_cat[cat])*cut_off)]
                    dic_train_cat[cat] = dic_train_cat[cat][int(len(dic_train_cat[cat])*cut_off):]

                if mode == 'train':
                    self.scene_paths = [item for sublist in dic_train_cat.values() for item in sublist]

                    print(f"Train set has {len(self.scene_paths)} samples")
                    categories = list(set([el.split("/")[-1].split("_")[0] for el in train_scenes]))
                    print(f"Categories in train set: {categories}")

                elif dataset_name == 'toys_seen':
                    self.scene_paths = [item for sublist in dic_seen_test_cat.values() for item in sublist]

                    print(f"SEEN Test set has {len(self.scene_paths)} samples")
                    categories = list(set([el.split("/")[-1].split("_")[0] for el in self.scene_paths]))
                    print(f"Categories in SEEN test set: {categories}")

                elif dataset_name == 'toys_unseen':
                    self.scene_paths = unseen_scenes

                    print(f"UNSEEN Test set has {len(self.scene_paths)} samples")
                    categories = list(set([el.split("/")[-1].split("_")[0] for el in self.scene_paths]))
                    print(f"Categories in UNSEEN test set: {categories}")

    

    def __len__(self):  
        return len(self.scene_paths)

    def get_image_tensor(self, path, size):
        img = Image.open(path).convert('RGB')
        trans = get_transform(size = size, normalize = True)
        img = trans(img)
        return img    
    
    def __getitem__(self, index):
        try:
            scene_path = self.scene_paths[index]
            rgb_images = sorted(glob.glob(scene_path+'/RGB/*'))

            with open(scene_path+'/scene3d.metadata.json','r') as f:
                json_dict = json.load(f)

            RT_matrices = [np.array(cam_pose['rotation']) for cam_pose in json_dict['camera']['poses']]
            K_matrices = [np.array(json_dict['camera']['K'])]*len(RT_matrices)

            view_indices = np.random.choice(len(rgb_images), self.num_views, replace = False)
            batch_rgb_images_path = np.array([rgb_images[i] for i in view_indices])
            batch_RT = [RT_matrices[i] for i in view_indices]
            batch_K = [K_matrices[i] for i in view_indices]
            batch_rgb = torch.stack([self.get_image_tensor(rgb_path, self.dimension) for rgb_path in batch_rgb_images_path])
            
            object_box = np.array([objs['bbox'] for objs in json_dict['objects']])
            object_label = np.array([1 if ("-" in os.path.basename(objs['path'])) else 0 for objs in json_dict['objects']])
            
            random_transformation = torch.eye(4)

            if self.random_scene_rotation_augmentation:
                r = torch.rand(1) * 2*np.pi
                random_rotation = torch.tensor([[np.cos(r), -np.sin(r)],[np.sin(r), np.cos(r)]], dtype=torch.float32)
                random_transformation[:2,:2] = random_rotation

            rot_batch_RT, rot_XYZ  = apply_random_rotation_in_space(torch.Tensor(batch_RT), torch.Tensor(object_box[:,0,:]), random_transformation)
            object_box[:,0,:] = rot_XYZ.numpy()
            
            targets = {'boxes':torch.Tensor(object_box).flatten(1,2), 'labels':torch.Tensor(object_label).long()}

            # Retrieving the type of each obj inside a list of strings
            # for a normal object 'norm' is used as a descriptor, unless the abbreviation of the type of anomaly
            anom_types = retrieve_anomaly_types(scene_path, self.dataset_name)

            return_dict = {
                'rgb_images':batch_rgb,
                'cam_RT' : rot_batch_RT,
                'cam_K' : np.array(batch_K),
                'targets' : targets,
                'scene_path' : scene_path,
                'anom_types' : anom_types
            }

            return return_dict
        
        except:
            print("Errors while loading data...")
            return None



def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    batch_rgb = torch.stack([data['rgb_images'] for data in batch]) 
    batch_RT = np.stack([data['cam_RT'] for data in batch]) 
    batch_K = np.stack([data['cam_K'] for data in batch]) 
    targets = [data['targets'] for data in batch]
    scene_path = [data['scene_path'] for data in batch]
    anom_types = [data['anom_types'] for data in batch]

    return_dict = {
        'rgb_images': batch_rgb,
        'cam_RT': batch_RT,
        'cam_K': batch_K,
        'targets': targets,
        'scene_path': scene_path,
        'anom_types': anom_types
    }

    return return_dict



def data_sampler(dataset, shuffle):
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)

def get_dataloaders(config, dataset_name='toys'):
    train_dataset = OddOneOutDataset(config, mode='train', dataset_name=dataset_name)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.dataset.batch_size,
        sampler=data_sampler(train_dataset, shuffle=True),
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=config.dataset.num_workers,
        #worker_init_fn=seed_worker,
    )

    if 'parts' in dataset_name:
        test_dataset = OddOneOutDataset(config, mode='test', dataset_name='parts')
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.dataset.batch_size,
            sampler=data_sampler(test_dataset, shuffle=True),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=config.dataset.num_workers,
            #worker_init_fn=seed_worker,
        )

        return train_loader, test_loader
    
    elif 'toys' in dataset_name:
        seen_test_dataset = OddOneOutDataset(config, mode ='test', dataset_name=f'{dataset_name}_seen')
        unseen_test_dataset = OddOneOutDataset(config, mode ='test', dataset_name=f'{dataset_name}_unseen')

        seen_test_loader = torch.utils.data.DataLoader(
            seen_test_dataset,
            batch_size=config.dataset.batch_size,
            sampler=data_sampler(seen_test_dataset, shuffle=True),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=config.dataset.num_workers,
            #worker_init_fn=seed_worker,
        )

        unseen_test_loader = torch.utils.data.DataLoader(
            unseen_test_dataset,
            batch_size=config.dataset.batch_size,
            sampler=data_sampler(unseen_test_dataset, shuffle=True),
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=config.dataset.num_workers,
            #worker_init_fn=seed_worker,
        )

        return train_loader, [seen_test_loader, unseen_test_loader]
    



def get_AnomalyTesting_dataloader(config, anomaly_type, dataset_name='toys'):

    dataset_anomaly = OddOneOutDataset(config, dataset_name=dataset_name, test_anomaly_type=anomaly_type)

    anomaly_loader = torch.utils.data.DataLoader(
        dataset_anomaly,
        batch_size = config.dataset.batch_size,
        sampler=data_sampler(dataset_anomaly, shuffle=True),
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=config.dataset.num_workers,
        #worker_init_fn=seed_worker,
    )

    return anomaly_loader


def get_CountTesting_dataloader(config, count_obj, dataset_name='toys'):

    dataset_anomaly = OddOneOutDataset(config, dataset_name=dataset_name, count_obj=count_obj)

    anomaly_loader = torch.utils.data.DataLoader(
        dataset_anomaly,
        batch_size = config.dataset.batch_size,
        sampler=data_sampler(dataset_anomaly, shuffle=True),
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=config.dataset.num_workers,
        #worker_init_fn=seed_worker,
    )

    return anomaly_loader


def contain_anomaly(scene_folder, searched_anomaly_type, dataset_str):
    scene2d_path = os.path.join(scene_folder, "scene2d.metadata.json")
    
    try:
        with open(scene2d_path, 'r') as f:
            scene2d_data = json.load(f)          
            
            blend_paths = scene2d_data.get("blend_paths", []) # Process blend paths to find anomalies and the total number of objects in the scene
            unique_blend_names = set()
            
            for path in blend_paths:
                # Extract the last part of the path
                blend_name = os.path.basename(path)
                unique_blend_names.add(blend_name)


            for blend_name in unique_blend_names:
                not_consider = (scene_folder.split('/')[-1].split('-')[0] + ".blend") if dataset_str=='toys' else (scene_folder.split('/')[-1].split('-')[0] + ".obj")
                
                if blend_name != not_consider:
                    anomaly_type = blend_name.split('-')[1] # extract anomaly type from the blend name
                    if anomaly_type == searched_anomaly_type:
                        return True # in case the scene contains at least one instance of the requested anomaly type

            return False # the requested anomaly type has not been found inside the scene

    except (FileNotFoundError, json.JSONDecodeError) as err:
        return
    

def contain_n_obects(scene_folder, n_obj=(3, 4)):
    scene2d_path = os.path.join(scene_folder, "scene2d.metadata.json")
    
    try:
        with open(scene2d_path, 'r') as f:
            scene2d_data = json.load(f)              
            
            blend_paths = scene2d_data.get("blend_paths", [])  # blend_paths is the list containing all the objects inside the scene
            if len(blend_paths) == n_obj[0] or len(blend_paths) == n_obj[1]:
                return True # in case the scene contains N or M objects
            else:
                return False

    except (FileNotFoundError, json.JSONDecodeError) as err:
        return False
    

# Used for retrieving the type of objects inside a scene (normal or anomalous & in case specify the anomaly)
def retrieve_anomaly_types(scene_folder, dataset_str):
    scene2d_path = os.path.join(scene_folder, "scene2d.metadata.json")
    
    try:
        with open(scene2d_path, 'r') as f:
            scene2d_data = json.load(f)             
            
            blend_paths = scene2d_data.get("blend_paths", [])  # Process blend paths to find anomalies and the total number of objects in the scene
            not_consider = (scene_folder.split('/')[-1].split('-')[0] + ".blend") if 'toys' in dataset_str else (scene_folder.split('/')[-1].split('-')[0] + ".obj")
            object_types = []

            for path in blend_paths:
                # Extract the last part of the path
                blend_name = os.path.basename(path)

                if blend_name == not_consider: # in case the name of the 3d object file is equal to the one representing a normal one, is a normal objects (the anomalies contain a descriptive string in their name)
                    object_types.append("norm")
                else:
                    object_types.append(blend_name.split('-')[1]) # otherwise append the string describing the anomaly

            return object_types


    except (FileNotFoundError, json.JSONDecodeError) as err:
        return False