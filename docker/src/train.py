import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
from pycocotools.coco import COCO
import cv2

# Define a custom Dataset class for COCO
class SolarPanelDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.image_dir, image_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Generate mask
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann:
                mask = np.maximum(mask, self.coco.annToMask(ann))

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Normalize mask to 0 and 1
        mask = (mask > 0).float()

        return image, mask

# Define training and validation data transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Define data directories
data_dir = "/workspace/data/coco"
train_data_dir = os.path.join(data_dir, "train")
valid_data_dir = os.path.join(data_dir, "valid")
train_annotation_file = os.path.join(train_data_dir, "_annotations.coco.json")
valid_annotation_file = os.path.join(valid_data_dir, "_annotations.coco.json")

# Create datasets and dataloaders
train_dataset = SolarPanelDataset(image_dir=train_data_dir, annotation_file=train_annotation_file, transform=train_transform)
valid_dataset = SolarPanelDataset(image_dir=valid_data_dir, annotation_file=valid_annotation_file, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=8)

# Define model, loss, and optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Using Unet with a pretrained encoder
model = smp.Unet(
    encoder_name="resnet34",        # Choose encoder, e.g., resnet34
    encoder_weights="imagenet",    # Use ImageNet pre-trained weights
    in_channels=3,                  # Input channels (RGB)
    classes=1                       # Output channels (number of classes)
).to(DEVICE)

# Define loss function and optimizer
loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training function
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Validation function
def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Training loop
num_epochs = 100
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    valid_loss = validate_epoch(model, valid_loader, loss_fn, DEVICE)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # Save the model if validation loss improves
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

