from collections import namedtuple
import open3d.ml.torch as ml3d
import numpy as np
import os
from nuscenes import NuScenes
import torch
from open3d.ml.torch.ops import voxelize
from open3d.ml.torch.ops import knn_search


# Define a namedtuple for KPFCNN inputs
KPFInput = namedtuple('KPFInput', ['features', 'points', 'batch', 'neighbors'])

class CustomNuScenes(ml3d.datasets.NuScenes):
    def __init__(self, dataset_path, info_path=None, use_cache=False, version="v1.0-mini", first_subsampling_dl=0.1, num_kernel_points=15, **kwargs):
        """Initialize the custom NuScenes dataset with reduced semantic labels."""
        super().__init__(dataset_path=dataset_path, info_path=info_path, use_cache=use_cache, **kwargs)
        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=False)
        self.first_subsampling_dl = first_subsampling_dl
        self.num_kernel_points = num_kernel_points 

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
        """Data preprocessing for KPFCNN, including subsampling and batch indexing."""
        points = np.array(data['point'], dtype=np.float32)
        labels = np.array(data['label'], dtype=np.int32).reshape((-1,))
        feat = np.array(data['feat'], dtype=np.float32) if 'feat' in data and data['feat'] is not None else None

        # Define voxelization parameters
        voxel_size = self.first_subsampling_dl
        row_splits = torch.tensor([0, points.shape[0]], dtype=torch.int64)  # Single batch
        points_range_min = torch.tensor(points.min(axis=0), dtype=torch.float32)
        points_range_max = torch.tensor(points.max(axis=0), dtype=torch.float32)

        # Perform voxelization
        voxel_grid = voxelize(
            torch.tensor(points, dtype=torch.float32),
            row_splits=row_splits,
            voxel_size=torch.tensor([voxel_size, voxel_size, voxel_size], dtype=torch.float32),
            points_range_min=points_range_min,
            points_range_max=points_range_max,
        )

        # Extract subsampled indices
        indices = voxel_grid.voxel_point_indices.numpy()

        # Subsample points, features, and labels
        sub_points = points[indices]
        sub_labels = labels[indices]
        sub_feat = feat[indices] if feat is not None else None

        # Add neighbors for KPConv
        layer_radii = [self.first_subsampling_dl * (2**i) for i in range(5)]  # Example radii per KPConv layer
        neighbors = []
        for radius in layer_radii:
            # Compute row splits
            points_row_splits = torch.tensor([0, len(sub_points)], dtype=torch.int64)
            queries_row_splits = torch.tensor([0, len(sub_points)], dtype=torch.int64)

            # Compute neighbors for the given radius
            knn_result = knn_search(
                torch.tensor(sub_points),          # Queries (points to find neighbors for)
                torch.tensor(sub_points),          # Points to search for neighbors in
                k=self.num_kernel_points,          # Number of neighbors (set to a reasonable value, e.g., 16)
                points_row_splits=points_row_splits,
                queries_row_splits=queries_row_splits,
                metric="L2",                       # Use L2 distance metric
                ignore_query_point=False,          # Include the query point itself in the neighbors
                return_distances=False             # Neighbors only; no need for distances
            )

            # Add the computed neighbors to the list
            neighbors.append(knn_result.neighbors_index.reshape(sub_points.shape[0],-1).numpy())

        # Prepare final processed data
        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['batch'] = np.zeros(sub_points.shape[0], dtype=np.int64)  # All points belong to a single batch
        data['neighbors'] = neighbors  # Add neighbors to the data dictionary

        return data

    def transform(self, data, attr):
        """Convert numpy arrays to torch Tensors for KPFCNN input."""
        points = torch.from_numpy(data['point']).float()
        features = (
            torch.from_numpy(data['feat']).float()
            if data['feat'] is not None
            else torch.zeros(points.shape[0], dtype=torch.float32)
        )
        labels = torch.from_numpy(data['label']).long()
        batch_indices = torch.from_numpy(data['batch']).long()
        neighbors = [torch.from_numpy(neigh).long() for neigh in data['neighbors']]  # Convert neighbors to Tensors

        return KPFInput(features=features, points=points, batch=batch_indices, neighbors=neighbors), labels



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
        """Retrieve and process a single sample for DataLoader."""
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        pc = self.dataset.read_lidar(lidar_path)
        labels = self.dataset.read_semantic_labels(info)

        # Prepare data dictionary for KPFCNN format
        data = {
            'point': pc[:, :3],  # Use only XYZ coordinates
            'feat': pc[:, 3:],  # Use additional features (e.g., intensity)
            'label': labels
        }
        processed_data = self.dataset.preprocess(data, {'split': self.split})
        return self.dataset.transform(processed_data, {'split': self.split})
