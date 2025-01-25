import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Helper function to split the bdappv dataset into train and valid directories
def split_bdappv_dataset(img_dir, mask_dir, output_dir, split_ratio=0.3):
    train_img_dir = os.path.join(output_dir, "train/images")
    train_mask_dir = os.path.join(output_dir, "train/masks")
    valid_img_dir = os.path.join(output_dir, "valid/images")
    valid_mask_dir = os.path.join(output_dir, "valid/masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_mask_dir, exist_ok=True)

    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    img_train, img_valid, mask_train, mask_valid = train_test_split(
        img_files, mask_files, test_size=split_ratio, random_state=42
    )

    for file in img_train:
        shutil.copy(os.path.join(img_dir, file), train_img_dir)
    for file in mask_train:
        shutil.copy(os.path.join(mask_dir, file), train_mask_dir)
    for file in img_valid:
        shutil.copy(os.path.join(img_dir, file), valid_img_dir)
    for file in mask_valid:
        shutil.copy(os.path.join(mask_dir, file), valid_mask_dir)

    print("bdappv dataset splitting completed!")


# Custom dataset class for semantic segmentation
class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

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
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Directories
data_dir = "/workspace/data"
train_data_dir = os.path.join(data_dir, "train")
valid_data_dir = os.path.join(data_dir, "valid")

# Split bdappv dataset
bdappv_data_dir = "/workspace/data/bdappv/google"
bdappv_img_dir = os.path.join(bdappv_data_dir, "img")
bdappv_mask_dir = os.path.join(bdappv_data_dir, "mask")
split_bdappv_dataset(bdappv_img_dir, bdappv_mask_dir, data_dir, split_ratio=0.3)

# Datasets
train_dataset_1 = SemanticSegmentationDataset(
    image_dir=os.path.join(train_data_dir, "images"),
    mask_dir=os.path.join(train_data_dir, "masks"),
    transform=train_transform
)

valid_dataset_1 = SemanticSegmentationDataset(
    image_dir=os.path.join(valid_data_dir, "images"),
    mask_dir=os.path.join(valid_data_dir, "masks"),
    transform=valid_transform
)

train_dataset_2 = SemanticSegmentationDataset(
    image_dir=os.path.join(data_dir, "train/images"),
    mask_dir=os.path.join(data_dir, "train/masks"),
    transform=train_transform
)

valid_dataset_2 = SemanticSegmentationDataset(
    image_dir=os.path.join(data_dir, "valid/images"),
    mask_dir=os.path.join(data_dir, "valid/masks"),
    transform=valid_transform
)

# Combine datasets
combined_train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
combined_valid_dataset = ConcatDataset([valid_dataset_1, valid_dataset_2])

# Data loaders
train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True, num_workers=16)
valid_loader = DataLoader(combined_valid_dataset, batch_size=64, shuffle=False, num_workers=16)

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
num_epochs = 20
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    valid_loss = validate_epoch(model, valid_loader, loss_fn, DEVICE)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

