import open3d as o3d
import numpy as np
import torch
import torch.nn as nn

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
    
import numpy as np

def compute_mIoU(pred, label, num_classes, per_class=False):
    """
    Compute mean Intersection over Union (mIoU) for a batch, 
    optionally returning per-class IoUs.

    Args:
        pred (np.ndarray): Predicted labels (N,).
        label (np.ndarray): Ground truth labels (N,).
        num_classes (int): Total number of classes.
        per_class (bool): If True, also returns per-class IoUs.

    Returns:
        float: mIoU (mean IoU across all classes).
        dict (optional): Per-class IoUs (only if per_class=True).
    """
    iou_list = []
    per_class_iou = {}

    pred = pred.flatten()
    label = label.flatten()

    for cls in range(num_classes):
        intersection = np.sum((pred == cls) & (label == cls))
        union = np.sum((pred == cls) | (label == cls))
        if union == 0:
            iou = 1  # Perfect IoU for empty class
        else:
            iou = intersection / union
        iou_list.append(iou)
        
        if per_class:
            per_class_iou[cls] = iou

    mIoU = np.mean(iou_list)
    
    if per_class:
        return mIoU, per_class_iou
    return mIoU

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        """
        Focal Loss with optional class weights.

        Args:
            alpha (torch.Tensor or float): Per-class weights (vector) or scalar.
            gamma (float): Focusing parameter.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")  # Per-sample loss

    def forward(self, logits, targets):
        """
        Compute Focal Loss.

        Args:
            logits (torch.Tensor): Predicted logits, shape (batch_size, num_classes, ...).
            targets (torch.Tensor): True labels, shape (batch_size, ...).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Compute per-sample Cross-Entropy Loss
        ce_loss = self.ce_loss(logits, targets)  # Shape: (batch_size, ...)

        # Compute pt (probability of the true class)
        pt = torch.exp(-ce_loss)  # Shape: (batch_size, ...)

        # Apply the focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]  # Get weights for each target class
                focal_loss = alpha_t * focal_loss
            else:  # Scalar alpha
                focal_loss = self.alpha * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss