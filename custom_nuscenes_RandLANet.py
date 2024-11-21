from collections import namedtuple
import open3d.ml.torch as ml3d
import numpy as np
import os
from nuscenes import NuScenes
import torch
from sklearn.neighbors import KDTree
from open3d._ml3d.datasets.utils import DataProcessing
# import random
# np.random.seed(42)
# torch.manual_seed(42)
# random.seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

class CustomNuScenes(ml3d.datasets.NuScenes):
    def __init__(self, dataset_path, info_path=None, use_cache=False, cache_dir=".", version="v1.0-mini", 
                 first_subsampling_dl=0.1, num_neighbors=16, num_layers=4, sub_sampling_ratio=[4, 4, 4, 4], **kwargs):
        """Initialize the custom NuScenes dataset."""
        super().__init__(dataset_path=dataset_path, info_path=info_path, use_cache=use_cache, cache_dir=cache_dir, **kwargs)
        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=False)
        self.first_subsampling_dl = first_subsampling_dl
        self.num_neighbors = num_neighbors
        self.num_layers = num_layers
        self.sub_sampling_ratio = sub_sampling_ratio

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
        lidarseg_record = self.nusc.get('lidarseg', lidar_token)
        lidarseg_filepath = os.path.join(self.dataset_path, lidarseg_record['filename'])
        semantic_labels = np.fromfile(lidarseg_filepath, dtype=np.uint8)
        remapped_labels = np.vectorize(self.map_to_new_class)(semantic_labels)
        return remapped_labels

    def preprocess(self, data, attr):
        """Preprocess data for RandLANet."""
        points = np.array(data['point'], dtype=np.float32)
        labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        feat = np.array(data['feat'], dtype=np.float32) if 'feat' in data and data['feat'] is not None else None

        # Step 1: Restrict points to be within a specific distance range in BEV (XY plane)
        min_bev_radius = 3.0   # Minimum radius to exclude points too close (e.g., hitting the vehicle)
        max_bev_radius = 40.0  # Maximum radius to exclude far-away points
        distances = np.sqrt(np.sum(points[:, :2] ** 2, axis=1))  # Compute distances in the XY plane
        mask = (distances >= min_bev_radius) & (distances <= max_bev_radius)  # Mask for points in the desired range
        points = points[mask]  # Apply the mask to the points
        labels = labels[mask]  # Apply the mask to the labels
        if feat is not None:
            feat = feat[mask]  # Apply the mask to the features if they exist

        # Step 2: Perform grid subsampling
        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=self.first_subsampling_dl)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=self.first_subsampling_dl)

        # Step 3: Create KDTree for nearest neighbor search
        search_tree = KDTree(sub_points)
        proj_inds = np.squeeze(
            search_tree.query(points, return_distance=False)).astype(np.int32)

        # Step 4: Pack the preprocessed data
        data = {
            'point': sub_points,
            'feat': sub_feat,
            'label': sub_labels,
            'search_tree': search_tree,
            'proj_inds': proj_inds
        }
        return data


    def transform(self, data, attr):
        """Transform data into RandLANet-compatible input with hierarchical downsampling."""
        target_num_points = 4096 * 2 # Number of points for the first layer

        # Step 1: Prepare the initial point cloud
        points = data['point']  # Original coordinates (XYZ)
        labels = data['label']
        features = data['feat'] if data['feat'] is not None else points.copy()

        # Normalize intensity (assume it's the first feature in `features`)
        if features is not None:
            intensity = features[:, 0]  # Extract intensity
            normalized_intensity = (intensity - np.mean(intensity)) / (np.std(intensity) + 1e-8)
            features[:, 0] = normalized_intensity  # Replace the intensity with the normalized version

        # Normalize coordinates for features only
        normalized_coords = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-8)
        # features = np.concatenate([features, normalized_coords], axis=1)  # Append normalized coords to features
        features = normalized_coords

        # Ensure the first layer has exactly `target_num_points`
        if points.shape[0] > target_num_points:
            indices = np.random.choice(points.shape[0], target_num_points, replace=False)
            points = points[indices]
            labels = labels[indices]
            features = features[indices]
        elif points.shape[0] < target_num_points:
            pad_size = target_num_points - points.shape[0]
            points = np.pad(points, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            labels = np.pad(labels, (0, pad_size), mode='constant', constant_values=0)
            features = np.pad(features, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

        # Step 2: Prepare hierarchical inputs
        input_points, input_neighbors, input_pools, input_up_samples = [], [], [], []
        for i in range(self.num_layers):
            # Compute nearest neighbors
            neighbor_idx = DataProcessing.knn_search(points, points, self.num_neighbors)

            # Subsample points for the next layer
            sub_points = points[:points.shape[0] // self.sub_sampling_ratio[i], :]
            pool_idx = neighbor_idx[:points.shape[0] // self.sub_sampling_ratio[i], :]
            up_idx = DataProcessing.knn_search(sub_points, points, 1)

            # Append data for the current layer
            input_points.append(torch.tensor(points, dtype=torch.float32))  # Current layer points
            input_neighbors.append(torch.tensor(neighbor_idx, dtype=torch.int64))  # Neighbors
            input_pools.append(torch.tensor(pool_idx, dtype=torch.int64))  # Downsampling indices
            input_up_samples.append(torch.tensor(up_idx, dtype=torch.int64))  # Upsampling indices

            # Update `points` for the next layer
            points = sub_points
            
        

        # Step 3: Return hierarchical inputs
        return {
            'coords': input_points,               # List of tensors (original XYZ coordinates)
            'neighbor_indices': input_neighbors,  # List of tensors
            'sub_idx': input_pools,               # List of tensors
            'interp_idx': input_up_samples,       # List of tensors
            'features': torch.tensor(features, dtype=torch.float32),  # Tensor with normalized intensity + normalized coords
            'labels': torch.tensor(labels, dtype=torch.int64),        # Tensor
            'filename': data['filename']  # Original filename
        }





    def get_split(self, split):
        """Return a dataset split."""
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
        """Retrieve and process a single sample."""
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        pc = self.dataset.read_lidar(lidar_path)
        labels = self.dataset.read_semantic_labels(info)

        # Prepare data dictionary
        data = {
            'point': pc[:, :3],  # Use XYZ coordinates
            'feat': pc[:, 3:-1],  # Use additional features (e.g., intensity)
            'label': labels
        }
        processed_data = self.dataset.preprocess(data, {'split': self.split})
        processed_data['filename'] = lidar_path
        return self.dataset.transform(processed_data, {'split': self.split})
