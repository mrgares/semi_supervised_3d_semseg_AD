# Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from open3d.ml.torch.models import RandLANet
from custom_nuscenes_RandLANet import CustomNuScenes
import os
import open3d as o3d
import helpers
import numpy as np
import open3d.ml.torch as ml3d
from open3d.ml.torch.vis import LabelLUT
from itertools import islice

from rich.traceback import install
install()

### General Parameters ###
i = 10 # Index of the batch you want
DATASET_ROOT = "/datastore/nuScenes/"
PRETRAINED_WEIGHTS_PATH = "./3d_semseg_model_randlanet_baseline.pth"
VERSION = "v1.0-mini"
BATCH_SIZE = 8
NUM_OF_WORKERS = 8
### Model Parameters ###
NUM_CLASSES = 3
NUM_NEIGHBORS = 16
NUM_LAYERS = 4
SUB_SAMPLING_RATIO = [4, 4, 4, 4]
DIM_FEATURES = 8
DIM_OUTPUT = [16, 64, 128, 256]
FIRST_SUBSAMPLING_DL = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
##### Main Logic #####
######################

# Check if dataset root exists
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

# Initialize the RandLANet model
model = RandLANet(
    name="RandLANet",
    num_neighbors=NUM_NEIGHBORS,
    num_layers=NUM_LAYERS,
    num_points=4096 * 2 * BATCH_SIZE,  # Based on batch size and expected points per sample
    num_classes=NUM_CLASSES,
    ignored_label_inds=[],  # Ignore class 0 (e.g., background)
    sub_sampling_ratio=SUB_SAMPLING_RATIO,
    in_channels=3,  # Number of input feature dimensions (e.g., intensity)
    dim_features=DIM_FEATURES,
    dim_output=DIM_OUTPUT,
    grid_size=FIRST_SUBSAMPLING_DL,
    num_workers = NUM_OF_WORKERS,
).to(DEVICE)
model.device = DEVICE

# Modify the classification head if needed
model.fc1[-1] = torch.nn.Conv2d(32, NUM_CLASSES, kernel_size=1, bias=False).to(DEVICE)

# Load pretrained weights
pretrained_weights = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)

# Load the pretrained weights into the model
model.load_state_dict(pretrained_weights)
print("Pretrained weights loaded successfully.")

# Initialize the custom dataset and DataLoader
dataset = CustomNuScenes(
    dataset_path=DATASET_ROOT,
    info_path=os.path.join(DATASET_ROOT, VERSION),
    use_cache=True,
    cache_dir="./cache",
    version=VERSION,
    first_subsampling_dl=FIRST_SUBSAMPLING_DL,  # Match grid size to model
    num_neighbors=NUM_NEIGHBORS,               # Match neighbor count
    num_layers=NUM_LAYERS,                     # Match number of layers
    sub_sampling_ratio=SUB_SAMPLING_RATIO,     # Match downsampling ratio
)

label_to_names_dict = dataset.label_to_names

# val_split = dataset.get_split("val")
val_split = dataset.get_split("val")
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=NUM_OF_WORKERS, pin_memory=True)# collate_fn=helpers.randlanet_collate_fn)
# batch = next(iter(val_loader))
batch = next(islice(val_loader, i, i+1))

model.eval()
with torch.no_grad(): # Disable gradient tracking
    batch = helpers.send_to_device(batch, DEVICE)
    predictions = model(batch)
    coords = batch['coords'][0].cpu().numpy()[0]
    labels = batch['labels'].cpu().numpy()[0]
    preds = torch.argmax(predictions, dim=-1)
    preds = preds[0].cpu().numpy()
    filename = batch['filename'][0]
print(f"Loaded point cloud: {filename}")

pred_counts = np.bincount(preds.flatten(), minlength=NUM_CLASSES)
label_counts = np.bincount(labels.flatten(), minlength=NUM_CLASSES)

print(f"Pred counts: {pred_counts}")  # Ensures all classes are included
print(f"Label counts: {label_counts}")

label_to_color = {
    0: [0.0, 0.0, 0.0],   # Black for label 0 (background)
    1: [0.0, 1.0, 1.0],   # Cyan for label 1 (human)
    2: [1.0, 0.0, 0.0]    # Red for label 2 (vehicle)
}

label_to_color_pred = {
    0: [0.0, 0.0, 0.0],   # Black for label 0 (background)
    1: [0.0, 0.0, 1.0],   # Blue for label 1 (human)
    2: [1.0, 0.0, 1.0]    # Purple for label 2 (vehicle)
}

colors = np.array([label_to_color[label] for label in labels])
colors_pred = np.array([label_to_color[label] for label in preds])
# copy coords to apply different colors
coords_pred = coords.copy()

# Create Open3D Point Cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.colors = o3d.utility.Vector3dVector(colors)

pcd_pred = o3d.geometry.PointCloud()
pcd_pred.points = o3d.utility.Vector3dVector(coords_pred)
pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
# Visualize the point cloud
vis = ml3d.vis.Visualizer()

offset = 60  # Distance to shift the second point cloud
coords2_shifted = coords_pred + np.array([offset, 0, 0])  # Shift along X-axis
# colors_pred_shifted = colors_pred 
vis_d = [
    {
        "name": "Ground Truth",  # Name of the first point cloud
        "points": coords,  # The coordinates of the point cloud
        "labels": labels,  # The ground truth labels for coloring
    },
    {
        "name": "Predictions",  # Name of the second point cloud
        "points": coords2_shifted,  # The same coordinates for the point cloud
        "labels": preds,   # The predicted labels for coloring
    },
    
]

lut = LabelLUT()
lut.add_label('background', 0, [0.0, 0.0, 0.0])  # Black
lut.add_label('human', 1, [0.0, 1.0, 1.0])       # Cyan
lut.add_label('vehicle', 2, [1.0, 0.0, 0.0])     # Red
vis.visualize(vis_d, lut=lut)
    

# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([pcd_pred])

