import open3d.ml.torch as ml3d
import numpy as np
import os
from nuscenes import NuScenes 
from torch import from_numpy
import torch
class CustomNuScenes(ml3d.datasets.NuScenes):
    def __init__(self, dataset_path, info_path=None, use_cache=False, version="v1.0-mini", **kwargs):
        """Initialize the custom NuScenes dataset with reduced semantic labels."""
        super().__init__(dataset_path=dataset_path, info_path=info_path, use_cache=use_cache, **kwargs)
        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=False)

    @staticmethod
    def get_label_to_names():
        """Define the mapping from original labels to the new classes for segmentation."""
        label_to_names = {
            0: 'background',  # Ignored classes
            1: 'human',       # Grouped as human
            2: 'vehicle'      # Grouped as vehicle
        }
        return label_to_names

    def map_to_new_class(self, original_label):
        """Map original NuScenes class names to new grouped segmentation labels."""
        human_classes = [2, 3, 4, 5, 6, 7, 8]  # Indices for human-related classes
        vehicle_classes = [14, 15, 16, 17, 18, 20, 21, 22, 23]  # Indices for vehicle-related classes

        if original_label in human_classes:
            return 1  # human
        elif original_label in vehicle_classes:
            return 2  # vehicle
        else:
            return 0  # background

    def read_semantic_labels(self, info):
        """Read and remap semantic labels for the point cloud."""
        sample = self.nusc.get('sample', info['token'])
        lidar_token = sample['data']['LIDAR_TOP']
        print(self.nusc.get('sample_data', lidar_token)['filename'])
        lidarseg_record = self.nusc.get('lidarseg', lidar_token)
        lidarseg_filepath = os.path.join(self.dataset_path, lidarseg_record['filename'])
        semantic_labels = np.fromfile(lidarseg_filepath, dtype=np.uint8)
        remapped_labels = np.vectorize(self.map_to_new_class)(semantic_labels)
        return remapped_labels

    def get_split(self, split):
        """Return a dataset split with remapped semantic labels."""
        return CustomNuScenesSplit(self, split=split)

class CustomNuScenesSplit:
    def __init__(self, dataset, split='train'):
        self.dataset = dataset
        self.split = split
        self.infos = dataset.get_split_list(split)

        if not self.infos:
            raise ValueError(f"No data found for the '{split}' split. Please check dataset paths and split names.")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        """Allow indexing to retrieve samples in DataLoader."""
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        pc = self.dataset.read_lidar(lidar_path)
        labels = self.dataset.read_semantic_labels(info)

        # Convert to tensors if not already
        pc_tensor = from_numpy(pc) if not isinstance(pc, torch.Tensor) else pc
        labels_tensor = from_numpy(labels) if not isinstance(labels, torch.Tensor) else labels

        # Return data, omitting 'feat' or setting it as an empty array
        data = {
            'point': pc_tensor,
            'feat': torch.empty((0,)),  # empty feature tensor if no features are used
            'label': labels_tensor
        }
        return data
