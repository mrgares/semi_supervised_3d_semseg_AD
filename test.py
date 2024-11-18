import torch
from open3d.ml.torch.models import PVCNN

# Parameters for the model and point cloud
num_points = 1024  # Number of points in the point cloud
num_classes = 13   # Number of output classes
num_features = 6   # Additional features (e.g., RGB)
voxel_resolution = 32  # Voxel grid resolution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_and_scale_coords(coords):
    """
    Normalize and scale coordinates to [0, 1].
    Args:
        coords: Tensor of shape (B, 3, N).
    Returns:
        Normalized and scaled coordinates.
    """
    coords_mean = coords.mean(dim=2, keepdim=True)
    coords_normalized = coords - coords_mean  # Center around 0
    coords_scaled = coords_normalized / (coords_normalized.abs().max() + 1e-5)  # Scale to [-1, 1]
    return (coords_scaled + 1) / 2  # Scale to [0, 1]

def test_pvcnn():
    # Generate random synthetic point cloud data
    point_cloud = torch.rand((1, 3, num_points), device=device)  # (Batch, Channels, Points)
    features = torch.rand((1, num_features, num_points), device=device)  # (Batch, Channels, Points)

    # Normalize and scale the point cloud coordinates
    normalized_coords = normalize_and_scale_coords(point_cloud)

    # Concatenate coordinates with features
    combined_features = torch.cat([normalized_coords, features], dim=1)  # Shape: (B, 9, N)

    # Prepare the input in the expected dictionary format
    inputs = {
        'point': normalized_coords,  # Scaled XYZ coordinates
        'feat': combined_features    # Combined features (XYZ + additional features)
    }

    # Define the PVCNN model
    model = PVCNN(
        name='PVCNN',
        device=device,
        num_classes=num_classes,
        extra_feature_channels=num_features,
        voxel_resolution_multiplier=voxel_resolution / 32,
    ).to(device)  # Move model to GPU

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass
    with torch.no_grad():
        output = model(inputs)

    # Print the output shape
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

if __name__ == "__main__":
    test_pvcnn()
