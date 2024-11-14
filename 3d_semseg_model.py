from torch.utils.data import DataLoader
import open3d.ml.torch as ml3d
import torch
import os
import helpers

# Import CustomNuScenes (assuming it's in the same file or accessible as a module)
from custom_nuscenes import CustomNuScenes

# Paths and parameters
DATASET_ROOT = "/datastore/nuScenes/"
VERSION = "v1.0-mini"

# Check if paths exist before proceeding
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

# Initialize the custom dataset
dataset = CustomNuScenes(dataset_path=DATASET_ROOT, info_path=DATASET_ROOT + 'v1.0-mini', use_cache=False)

# Use a DataLoader to load the data in batches
data_loader = DataLoader(dataset.get_split('train'), batch_size=1, shuffle=True, num_workers=4)

# Function to print out a sample to verify data loading and label mapping
def verify_sample():
    for idx, data in enumerate(data_loader):
        point_cloud = data['point']  # Point cloud data
        labels = data['label']       # Remapped semantic labels
        
        print(f"Sample {idx}")
        print("Point Cloud Shape:", point_cloud.shape)
        print("Labels Shape:", labels.shape)
        print("Unique Labels:", torch.unique(labels))
        helpers.visualize_sample(point_cloud, labels)
        # Break after one sample for verification purposes
        break

# Run verification
verify_sample()



# split = dataset.get_split_list('training') # could be 'training', 'validation', or 'testing'

# # print the attributes of the first datum
# print(split.get_attr(0))

# # print the shape of the first point cloud
# print(split.get_data(0)['point'].shape)

# # Show the first 340 frames using the Open3D visualizer
# vis = ml3d.vis.Visualizer()
# vis.visualize_dataset(dataset, 'all', indices=range(340))