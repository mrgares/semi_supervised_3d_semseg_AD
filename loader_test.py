from custom_nuscenes_RandLANet import CustomNuScenes
import os
import torch
from torch.utils.data import DataLoader
import helpers
import open3d as o3d
import numpy as np

# Parameters
DATASET_ROOT = "/datastore/nuScenes/"
VERSION = "v1.0-mini"
BATCH_SIZE = 5
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FL_ALPHA_PER_CLASS = [0.01, 0.495, 0.495]
FL_GAMMA = 2
FIRST_SUBSAMPLING_DL = 0.1
NUM_NEIGHBORS = 16
NUM_LAYERS = 4
SUB_SAMPLING_RATIO = [4, 4, 4, 4]
NUM_CLASSES = 3

# Initialize dataset and split
dataset = CustomNuScenes(
    dataset_path=DATASET_ROOT,
    info_path=os.path.join(DATASET_ROOT, VERSION),
    use_cache=False,
    version=VERSION,
    first_subsampling_dl=FIRST_SUBSAMPLING_DL,
    num_neighbors=NUM_NEIGHBORS,
    num_layers=NUM_LAYERS,
    sub_sampling_ratio=SUB_SAMPLING_RATIO,
)

train_split = dataset.get_split('train')

# Create DataLoader
train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)

### Visualize a batch of data ###

# Fetch a batch of data from the train_loader
batch = next(iter(train_loader))

# Extract point cloud coordinates and labels
coords = batch['coords'][0].numpy()[0]  # Use first layer (Layer 0) points
labels = batch['labels'].numpy()[0]  # Corresponding labels for points
filename = batch['filename'][0]  # File name of the point cloud
print(f"Loaded point cloud: {filename}")
print(f"Num of points: {coords.shape[0]}")
# Flatten the batch dimension
coords = coords.reshape(-1, 3)  # Flatten into N x 3
labels = labels.flatten()  # Flatten into N

# Normalize labels to fit Open3D color range [0, 1]
unique_labels = np.unique(labels)
label_to_color = {
    0: [0, 0, 0],       # Black for label 0 (background)
    1: [0, 1, 1],       # Cyan for label 1 (human)
    2: [1, 0, 0]        # Red for label 2 (vehicle)
}
colors = np.array([label_to_color[label] for label in labels])

# Create Open3D Point Cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud {filename}")
