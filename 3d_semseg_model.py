# Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from open3d.ml.torch.models import PVCNN
from custom_nuscenes import CustomNuScenes
import os
import torch.nn as nn

# Parameters
DATASET_ROOT = "/datastore/nuScenes/"
VERSION = "v1.0-mini"
BATCH_SIZE = 1                # Using batch size 1 for simplicity; adjust as needed.
NUM_EPOCHS = 10               # Number of epochs for training
LEARNING_RATE = 0.001         # Learning rate for the optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_loader = DataLoader(dataset.get_split('train'), batch_size=BATCH_SIZE, shuffle=True)

# Initialize the PVCNN model
model = PVCNN(num_classes=3,  # Assuming 3 classes based on your label mapping
              num_points=40960,  # Matching the point count in the loader
              extra_feature_channels=9,  # Includes x, y, z + extra features
              width_multiplier=1,
              voxel_resolution_multiplier=1).to(DEVICE)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
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
        loss = criterion(outputs.view(-1, 3), labels.view(-1))  # Reshape for compatibility
        loss.backward()
        
        # Update the model parameters
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
            running_loss = 0.0
            
print("Training completed.")