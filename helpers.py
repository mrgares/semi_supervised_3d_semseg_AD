import open3d as o3d
import numpy as np
import torch

def visualize_sample(point_cloud, labels):
    """
    Visualize a point cloud with color-coded segmentation labels.
    """
    # Convert point cloud and labels to numpy arrays if they are PyTorch tensors
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.squeeze(0).numpy()  # Remove batch dimension
    if isinstance(labels, torch.Tensor):
        labels = labels.squeeze(0).numpy()

    # Define colors for each label (adjust as desired)
    label_colors = {
        0: [0.5, 0.5, 0.5],  # Background - Gray
        1: [1, 0, 0],        # Human - Red
        2: [0, 0, 1]         # Vehicle - Blue
    }

    # Create an Open3D PointCloud object
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # Use only XYZ coordinates

    # Apply colors based on labels
    colors = np.array([label_colors[label] for label in labels])
    o3d_pc.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([o3d_pc])