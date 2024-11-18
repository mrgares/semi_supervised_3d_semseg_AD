# Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from open3d.ml.torch.models import KPFCNN
from custom_nuscenes_KPFCNN import CustomNuScenes
import os
import wandb
import helpers
import numpy as np
from custom_nuscenes_KPFCNN import KPFInput

from rich.traceback import install
install()


# Parameters
DATASET_ROOT = "/datastore/nuScenes/"
VERSION = "v1.0-mini"
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FL_ALPHA_PER_CLASS = [0.01, 0.495, 0.495]
FL_GAMMA = 2
FIRST_SUBSAMPLING_DL = 0.2
NUM_KERNEL_POINTS=3

# Check if dataset root exists
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

# Initialize the KPFCNN model
model = KPFCNN(
    name='KPFCNN',
    num_classes=3,
    first_subsampling_dl=FIRST_SUBSAMPLING_DL,
    in_features_dim=2,  # Additional feature dimensions
    num_layers=4,
    num_kernel_points=NUM_KERNEL_POINTS,
    first_features_dim=64,
    batch_norm_momentum=0.02,
).to(DEVICE)



# Initialize the custom dataset and DataLoader
dataset = CustomNuScenes(
    dataset_path=DATASET_ROOT,
    info_path=os.path.join(DATASET_ROOT, VERSION),
    use_cache=False,
    version=VERSION,
    first_subsampling_dl=FIRST_SUBSAMPLING_DL,
    num_kernel_points=model.K, # Number of kernel points
)
label_to_names_dict = dataset.label_to_names
train_split = dataset.get_split("train")
train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=helpers.custom_collate_fn)
val_loader = DataLoader(dataset.get_split('val'), batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=helpers.custom_collate_fn)

# Define loss function and optimizer
class_weights = torch.tensor(FL_ALPHA_PER_CLASS).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize WandB
wandb.init(
    project="3d_semseg_project",  
    config={  # Log hyperparameters
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model": "KPFCNN",
    },
    name=f"KPFCNN_bs:{BATCH_SIZE}_lr:{LEARNING_RATE}_ep:{NUM_EPOCHS}_cel_class_weights:{FL_ALPHA_PER_CLASS}"
)

# Track the model in WandB
wandb.watch(model, log="all")

# Training and Validation loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    # Training loop
    for i, (inputs, labels) in enumerate(train_loader):
        # Move batch data to device
        inputs = KPFInput(
            features=inputs.features.to(DEVICE),
            points=inputs.points.to(DEVICE),
            batch=inputs.batch.to(DEVICE),
            neighbors=inputs.neighbors.to(DEVICE) 
        )
        labels = labels.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)


        # Compute the loss
        loss = criterion(outputs.view(-1, 3), labels.view(-1))
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
        break

    # Validation loop
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    break
    with torch.no_grad():  # Disable gradient computation
        for i, (inputs, labels) in enumerate(val_loader):  # Inputs are KPFInput, labels are ground truth
            # Move data to device
            inputs = KPFInput(
            features=inputs.features.to(DEVICE),
            points=inputs.points.to(DEVICE),
            batch=inputs.batch.to(DEVICE),
            neighbors=inputs.neighbors.to(DEVICE) 
            )
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            val_loss += loss.item()

            # Get predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Convert to NumPy for mIoU computation
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    # Compute mIoU and log results
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mIoU, per_class_iou = helpers.compute_mIoU(all_preds, all_labels, num_classes=3, per_class=True)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Validation Loss: {avg_val_loss:.4f}, mIoU: {mIoU:.4f}")
    wandb.log({"Loss/Validation": avg_val_loss, "Validation mIoU": mIoU})
    for cls, iou in per_class_iou.items():
        wandb.log({f"Class_{label_to_names_dict[cls]}_IoU": iou})

# Save the trained model
SAVE_PATH = "3d_semseg_model_kp.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")

# Log the model to WandB
wandb.save(SAVE_PATH)
