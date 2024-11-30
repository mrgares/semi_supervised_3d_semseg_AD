# Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from open3d.ml.torch.models import RandLANet
from custom_nuscenes_RandLANet import CustomNuScenes

import os
import wandb
import helpers
import numpy as np
import warnings
from rich.traceback import install

install()


warnings.filterwarnings(
    "ignore",
    message="MongoClient opened before fork. May not be entirely fork-safe",
    category=UserWarning,
    module="pymongo.mongo_client",
)


### General Parameters ###
DATASET_ROOT = "/datastore/nuScenes/"
PRETRAINED_WEIGHTS_PATH = "/workspace/pretrained_weights/randlanet_semantickitti_202201071330utc.pth"
VERSION = "v1.0-trainval"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3#0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_OF_WORKERS = 8
PSEUDO_LABEL_RATIO = 0.75
### Model Parameters ###
# Loss parameters
FL_ALPHA_PER_CLASS = [0.01, 0.6, 0.39]  # Focal loss parameters
FL_GAMMA = 2

# Shared parameters for dataset and model
FIRST_SUBSAMPLING_DL = 0.1  # Grid size for subsampling
NUM_NEIGHBORS = 32          # Number of nearest neighbors
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

# Load pretrained weights
pretrained_weights = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)

# Extract the model state dictionary from the pretrained weights
state_dict = pretrained_weights['model_state_dict']

# Check if the classification head needs adjustment
classification_head_key = 'fc1.3.conv.weight'  # Key for the classification head weights
if state_dict[classification_head_key].shape[0] != NUM_CLASSES:
    # freezed the pretrained weights except the classification head
    # for key in state_dict.keys():
    #     if 'fc1' not in key:
    #         state_dict[key].requires_grad = False
    print(f"Adjusting the classification head for {NUM_CLASSES} classes.")
    model.fc1[-1] = torch.nn.Conv2d(32, NUM_CLASSES, kernel_size=1, bias=False).to(DEVICE)

# Load the state dictionary into the model
try:
    model.load_state_dict(state_dict, strict=False)  # strict=False allows mismatched keys
    print("Pretrained weights loaded successfully.")
except Exception as e:
    print(f"Error loading pretrained weights: {e}")

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
train_split = dataset.get_split("train", pseudo_label_ratio=PSEUDO_LABEL_RATIO)
val_split = dataset.get_split("val")
train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,num_workers=NUM_OF_WORKERS, pin_memory=True) #collate_fn=helpers.randlanet_collate_fn)
val_loader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=NUM_OF_WORKERS, pin_memory=True)# collate_fn=helpers.randlanet_collate_fn)

# Define loss function and optimizer
class_weights = torch.tensor(FL_ALPHA_PER_CLASS)
criterion = helpers.SemSegLoss(weights=class_weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize WandB
wandb.init(
    project="3d_semseg_project",
    config={
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model": "RandLANet",
    },
    name=f"[pseudo labels:{PSEUDO_LABEL_RATIO*100}%] RandLANet_bs:{BATCH_SIZE}_lr:{LEARNING_RATE}_ep:{NUM_EPOCHS}_cel_class_weights:{class_weights}",
)

# Track the model in WandB
wandb.watch(model, log="all")

# Training and Validation loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    # Training loop
    for i, batch in enumerate(train_loader):
        
        batch = helpers.send_to_device(batch, DEVICE)
        labels = batch["labels"]
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch)

        # Compute the loss
        loss, filtered_labels, filtered_scores = model.get_loss(criterion, 
                                                                predictions, 
                                                                {"data": batch},
                                                                DEVICE)
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        if (i + 1) % 10 == 0:  # Log every 10 batches
            avg_loss = running_loss / 10
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
            wandb.log({"Loss/Training": avg_loss, "epoch": epoch + 1, "step": i + 1})
            running_loss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient tracking
        all_preds = []
        all_labels = []
        val_loss = 0.0

        for batch in val_loader:
            batch = helpers.send_to_device(batch, DEVICE)
            labels = batch["labels"]  # Shape: (B, 4096)

            # Forward pass
            predictions = model(batch)  # Shape: (B, 4096, 3)

            # Get predictions
            preds = torch.argmax(predictions, dim=-1)  # Shape: (B, 4096)
            # pred_counts = np.bincount(preds[0].cpu().numpy().flatten(), minlength=NUM_CLASSES)
            # print(f"preds: {pred_counts}")
            # label_counts = np.bincount(labels[0].cpu().numpy().flatten(), minlength=NUM_CLASSES)
            # print(f"labels: {label_counts}")
            all_preds.append(preds.cpu().numpy().flatten())  
            all_labels.append(labels.cpu().numpy().flatten())

            # Compute loss
            loss, _, _ = model.get_loss(criterion, predictions, {"data": batch}, DEVICE)
            val_loss += loss.item()

        # Combine all predictions and labels across batches
        all_preds = np.concatenate(all_preds, axis=0)  # Shape: (total_points,)
        all_labels = np.concatenate(all_labels, axis=0)  # Shape: (total_points,)

        # Compute mIoU and per-class IoUs
        mIoU, per_class_iou = helpers.compute_mIoU(all_preds, all_labels, num_classes=NUM_CLASSES, per_class=True)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}, mIoU: {mIoU:.4f}")
        wandb.log({"Loss/Validation": avg_val_loss, "Validation mIoU": mIoU})
        for cls, iou in per_class_iou.items():
            wandb.log({f"Class_{label_to_names_dict[cls]}_IoU": iou})

# Save the trained model
SAVE_PATH = "3d_semseg_model_randlanet.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# Log the model to WandB
wandb.save(SAVE_PATH)
