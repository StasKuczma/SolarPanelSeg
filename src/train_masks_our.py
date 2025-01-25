import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Custom dataset class for semantic segmentation
class SemanticSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        self.transform = transform

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("The number of images and masks does not match!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))  # Assuming masks are grayscale

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Normalize mask to 0 and 1
        mask = (mask > 0).float()

        return image, mask


# Data transformations
train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.2),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2()
])

# Directories
train_data_dir = "/workspace/data/train"
valid_data_dir = "/workspace/data/valid"

# Datasets
train_dataset = SemanticSegmentationDataset(data_dir=train_data_dir, transform=train_transform)
valid_dataset = SemanticSegmentationDataset(data_dir=valid_data_dir, transform=valid_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=16)

# Model, loss, and optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training and validation functions
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


# Training loop
num_epochs = 300
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    valid_loss = validate_epoch(model, valid_loader, loss_fn, DEVICE)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "./models/best_model.pth")
        print("Model saved!")

