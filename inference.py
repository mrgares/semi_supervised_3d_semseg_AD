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
from rich.traceback import install
install()

### General Parameters ###
DATASET_ROOT = "/datastore/nuScenes/"
PRETRAINED_WEIGHTS_PATH = "./3d_semseg_model_randlanet.pth"
VERSION = "v1.0-mini"
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5#0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_OF_WORKERS = 8
### Model Parameters ###
# Loss parameters
FL_ALPHA_PER_CLASS = [0.01, 0.495, 0.495]  # Focal loss parameters
FL_GAMMA = 2

# Shared parameters for dataset and model
FIRST_SUBSAMPLING_DL = 0.1  # Grid size for subsampling
NUM_NEIGHBORS = 16          # Number of nearest neighbors
NUM_LAYERS = 4              # Number of encoder/decoder layers
SUB_SAMPLING_RATIO = [4, 4, 4, 4]  # Downsampling ratio per layer
NUM_CLASSES = 3             # Number of semantic classes

# Feature dimensions for RandLANet
DIM_FEATURES = 8            # Initial feature dimension
DIM_OUTPUT = [16, 64, 128, 256]  # Feature dimensions for each layer

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
    num_points=4096 * BATCH_SIZE,  # Based on batch size and expected points per sample
    num_classes=NUM_CLASSES,
    ignored_label_inds=[0],  # Ignore class 0 (e.g., background)
    sub_sampling_ratio=SUB_SAMPLING_RATIO,
    in_channels=3,  # Number of input feature dimensions (e.g., intensity)
    dim_features=DIM_FEATURES,
    dim_output=DIM_OUTPUT,
    grid_size=FIRST_SUBSAMPLING_DL,
    num_workers = NUM_OF_WORKERS,
).to(DEVICE)
model.device = DEVICE

# Load pretrained weights
pretrained_weights = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)

# Load the pretrained weights into the model
model.load_state_dict(pretrained_weights, strict=False)
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
val_split = dataset.get_split("val")
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=NUM_OF_WORKERS, pin_memory=True)# collate_fn=helpers.randlanet_collate_fn)
batch = next(iter(val_loader))

model.eval()
with torch.no_grad(): # Disable gradient tracking
    batch = helpers.send_to_device(batch, DEVICE)
    predictions = model(batch)
    coords = batch['coords'][0].cpu().numpy()[0]
    labels = batch['labels'].cpu().numpy()[0]
    preds = torch.argmax(predictions, dim=-1)
    preds = preds[0].cpu().numpy().flatten()
    filename = batch['filename'][0]
print(f"Loaded point cloud: {filename}")
label_to_color = {
    0: [0, 0, 0],       # Black for label 0 (background)
    1: [0, 1, 1],       # Cyan for label 1 (human)
    2: [1, 0, 0]        # Red for label 2 (vehicle)
}
# green, blue, purple
label_to_color_pred = {
    0: [0, 1, 0],       # Green for label 0 (background)
    1: [0, 0, 1],       # Blue for label 1 (human)
    2: [1, 0, 1]        # Purple for label 2 (vehicle)
}

colors = np.array([label_to_color[label] for label in labels])
colors_pred = np.array([label_to_color_pred[label] for label in preds])

# Create Open3D Point Cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.colors = o3d.utility.Vector3dVector(colors)

pcd_pred = o3d.geometry.PointCloud()
pcd_pred.points = o3d.utility.Vector3dVector(coords)
pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

# Visualize the point cloud
vis = ml3d.vis.Visualizer()
# vis_d = [{
#     "name": "Point Cloud comparison",
#     "points": coords,
#     "labels": labels,
#     "pred": preds
# }]

vis_d = [
    {
        "name": "Ground Truth",  # Name of the first point cloud
        "points": coords,  # The coordinates of the point cloud
        "labels": labels,  # The ground truth labels for coloring
    },
    {
        "name": "Predictions",  # Name of the second point cloud
        "points": coords,  # The same coordinates for the point cloud
        "labels": preds,   # The predicted labels for coloring
    },
]
vis.visualize(vis_d)


    


