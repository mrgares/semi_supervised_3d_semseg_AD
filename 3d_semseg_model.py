# Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from open3d.ml.torch.models import PVCNN, KPFCNN
from custom_nuscenes import CustomNuScenes
import os
import torch.nn as nn
import wandb  
import helpers
import numpy as np

# Parameters
DATASET_ROOT = "/datastore/nuScenes/"
VERSION = "v1.0-mini"
BATCH_SIZE = 10
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FL_ALPHA_PER_CLASS = [0.01, 0.495, 0.495] #[0.0032, 0.9540, 0.0428]
FL_GAMMA = 2


# Check if dataset root exists
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

# Initialize the custom dataset and DataLoader
dataset = CustomNuScenes(
    dataset_path=DATASET_ROOT,
    info_path=os.path.join(DATASET_ROOT, VERSION),
    use_cache=False,
    version=VERSION,
    device=DEVICE
)
label_to_names_dict = dataset.label_to_names
train_split = dataset.get_split("train")
train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset.get_split('val'), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# Initialize the PVCNN model
model = PVCNN(num_classes=3,
              num_points=40960,
              extra_feature_channels=2, # 2 extra feature channels + 3 for XYZ
              width_multiplier=1,
              voxel_resolution_multiplier=1).to(DEVICE)

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()

class_weights = torch.tensor(FL_ALPHA_PER_CLASS).to(DEVICE)
# criterion = helpers.FocalLoss(alpha=class_weights, gamma=FL_GAMMA)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize WandB
wandb.init(
    project="3d_semseg_project",  
    config={  # Log hyperparameters
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model": "PVCNN",
    },
    # name=f"PVCNN_bs:{BATCH_SIZE}_lr:{LEARNING_RATE}_ep:{NUM_EPOCHS}_fl_alpha_per_class:{FL_ALPHA_PER_CLASS}fl_gamma:{FL_GAMMA}"
    name=f"PVCNN_bs:{BATCH_SIZE}_lr:{LEARNING_RATE}_ep:{NUM_EPOCHS}_cel_class_weights:{FL_ALPHA_PER_CLASS}"
)

# Track the model in WandB
wandb.watch(model, log="all")

# Training and Validation loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    # Training loop
    for i, batch in enumerate(train_loader):
        # Move batch data to device
        points = batch['point'].to(DEVICE)
        features = batch['feat'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Reshape to fit PVCNN input expectations
        points = points[:, :, :3].permute(0, 2, 1)  # Shape: (B, 3, N)
        features = features.permute(0, 2, 1)  # Shape: (B, C, N)

        # Forward pass
        outputs = model({'point': points, 'feat': features})

        # Compute the loss
        loss = criterion(outputs.reshape(-1, 3), labels.reshape(-1))
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
    with torch.no_grad():
        for batch in val_loader:
            # Move batch data to device
            points = batch['point'].to(DEVICE)
            features = batch['feat'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Reshape to fit PVCNN input expectations
            points = points[:, :, :3].permute(0, 2, 1)  # Shape: (B, 3, N)
            features = features.permute(0, 2, 1)  # Shape: (B, C, N)

            # Forward pass
            outputs = model({'point': points, 'feat': features})

            # Compute the loss
            loss = criterion(outputs.reshape(-1, 3), labels.reshape(-1))
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=2).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    # Compute mIoU
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute mIoU and per-class IoU
    mIoU, per_class_iou = helpers.compute_mIoU(all_preds, all_labels, num_classes=3, per_class=True)
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
    wandb.log({"Loss/Validation": avg_val_loss, "Validation mIoU": mIoU, "epoch": epoch + 1})
    for cls, iou in per_class_iou.items():
        # set the class name in the key
        wandb.log({f"Class_{label_to_names_dict[cls]}_IoU": iou, "epoch": epoch + 1})

# Save the trained model
SAVE_PATH = "3d_semseg_model.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# Log the model to WandB
wandb.save(SAVE_PATH)
