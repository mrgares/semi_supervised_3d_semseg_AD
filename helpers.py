import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from custom_nuscenes_KPFCNN import KPFInput

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

###########################################
#### helper functions for KPFCNN model ####
###########################################

        
def pad_tensor(tensor, max_size, padding_value=0):
    if len(tensor.shape) == 1:  # Handle 1D tensors
        padding_length = max_size - tensor.size(0)
        if padding_length < 0:
            raise RuntimeError(f"Invalid padding length: {padding_length}.")
        return torch.nn.functional.pad(tensor, (0, padding_length), value=padding_value)
    elif len(tensor.shape) == 2:  # Handle 2D tensors
        padding_length = max_size - tensor.size(0)
        if padding_length < 0:
            raise RuntimeError(f"Invalid padding length: {padding_length}.")
        return torch.nn.functional.pad(tensor, (0, 0, 0, padding_length), value=padding_value)
    else:
        raise ValueError(f"Tensor must be 1D or 2D, but got shape {tensor.shape}.")



def custom_collate_fn(batch):
    # Separate inputs and labels
    inputs, labels = zip(*batch)

    # Find the maximum number of points in the batch for padding
    max_points = max(inp.points.size(0) for inp in inputs)

    # Pad features and points (2D tensors) to the maximum size in the batch
    padded_features = torch.stack([pad_tensor(inp.features, max_points) for inp in inputs])
    padded_points = torch.stack([pad_tensor(inp.points, max_points) for inp in inputs])

    # Update batch indices for the entire batch
    updated_batches = []
    for i, inp in enumerate(inputs):
        batch_indices = inp.batch.clone()  # Clone to avoid modifying the original tensor
        batch_indices[batch_indices != -1] += i  # Increment batch indices by the sample index
        padded_batch = pad_tensor(batch_indices, max_points, padding_value=-1)
        updated_batches.append(padded_batch)

    padded_batch = torch.stack(updated_batches)

    # Handle neighbors (list of lists, padded to equal lengths)
    max_neighbors = max(len(inp.neighbors[0]) for inp in inputs)
    padded_neighbors = [
        torch.stack([pad_tensor(n.clone().detach().to(torch.long), max_neighbors, padding_value=-1) for n in inp.neighbors])
        for inp in inputs
    ]
    padded_neighbors = torch.stack(padded_neighbors)  # Stack to make neighbors a PyTorch tensor

    # Stack labels as tensors
    padded_labels = torch.stack([pad_tensor(label, max_points, padding_value=-1) for label in labels])

    # Return padded data as KPFInput and labels
    return (
        KPFInput(features=padded_features, points=padded_points, batch=padded_batch, neighbors=padded_neighbors),
        padded_labels
    )

###########################################
### helper functions for randlanet model ##
###########################################


def pad_to_max_length(tensors, max_length, pad_value=0):
    """Pad a list of tensors to the same length."""
    padded = []
    for tensor in tensors:
        pad_size = max_length - tensor.size(0)
        padded.append(torch.cat([tensor, torch.full((pad_size,) + tensor.shape[1:], pad_value)]))
    return torch.stack(padded)

def create_mask(tensors, max_length):
    """
    Create a mask for padded tensors.
    
    Args:
        tensors: List of tensors (variable-sized).
        max_length: Integer, the length to which tensors are padded.
    
    Returns:
        Tensor of shape (batch_size, max_length), with 1 for valid points and 0 for padded points.
    """
    masks = []
    for tensor in tensors:
        valid_size = tensor.size(0)
        mask = torch.cat([torch.ones(valid_size), torch.zeros(max_length - valid_size)])
        masks.append(mask)
    return torch.stack(masks)


def randlanet_collate_fn(batch):
    """Collate function to handle variable-sized point clouds with padding and masks."""
    batched_inputs = {}

    # Determine max number of points across the batch
    max_points = max(sample['coords'][0].size(0) for sample in batch)

    # Pad and stack hierarchical inputs
    for key in batch[0]:
        if isinstance(batch[0][key], list):  # Hierarchical inputs like 'coords'
            batched_inputs[key] = [
                pad_to_max_length(
                    [sample[key][i] for sample in batch], max_length=max_points
                )
                for i in range(len(batch[0][key]))
            ]
        elif key == 'features':  # Features are also padded
            max_features_points = max(sample[key].size(0) for sample in batch)
            batched_inputs[key] = pad_to_max_length(
                [sample[key] for sample in batch], max_length=max_features_points
            )
        elif key == 'labels':  # Labels are padded
            max_labels_points = max(sample[key].size(0) for sample in batch)
            batched_inputs[key] = pad_to_max_length(
                [sample[key] for sample in batch], max_length=max_labels_points
            )
        else:  # Handle other non-variable inputs
            batched_inputs[key] = torch.stack([sample[key] for sample in batch])

    # Create and add masks
    batched_inputs['masks'] = create_mask(
        [sample['coords'][0] for sample in batch], max_length=max_points
    )

    return batched_inputs

def send_to_device(batch, device):
    """
    Move all tensors in the batch to the specified device.
    
    Args:
        batch (dict): Batched data from the DataLoader.
        device (torch.device): Target device (e.g., 'cuda' or 'cpu').
    
    Returns:
        dict: Batch with all tensors moved to the target device.
    """
    for key, value in batch.items():
        if isinstance(value, list):  # Handle hierarchical inputs (lists of tensors)
            # if is string, then it is filename, so skip
            if isinstance(value[0], str):
                continue
            batch[key] = [v.to(device) for v in value]
        elif isinstance(value, torch.Tensor):  # Handle flat tensors
            batch[key] = value.to(device)
    return batch

class SemSegLoss:
    def __init__(self, weights=None):
        # Define weighted CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)

    def weighted_CrossEntropyLoss(self, predictions, labels):
        return self.criterion(predictions, labels)

