import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Helper function to clean the bdappv dataset by removing unmatched files
def clean_bdappv_dataset(img_dir, mask_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    mask_basenames = {os.path.splitext(f)[0] for f in mask_files}
    unmatched_img_files = [f for f in img_files if os.path.splitext(f)[0] not in mask_basenames]

    # Delete unmatched image files
    for file in unmatched_img_files:
        os.remove(os.path.join(img_dir, file))
        print(f"Deleted unmatched image file: {file}")

    print("Cleaning bdappv dataset completed!")


# Helper function to split the bdappv dataset into train and valid directories
def split_bdappv_dataset(img_dir, mask_dir, output_dir, split_ratio=0.3):
    train_img_dir = os.path.join(output_dir, "train")
    train_mask_dir = os.path.join(output_dir, "train")
    valid_img_dir = os.path.join(output_dir, "valid")
    valid_mask_dir = os.path.join(output_dir, "valid")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    # Ensure only matching files are used
    img_files = [f for f in img_files if os.path.splitext(f)[0] in [os.path.splitext(m)[0] for m in mask_files]]
    mask_files = [f for f in mask_files if os.path.splitext(f)[0] in [os.path.splitext(i)[0] for i in img_files]]

    img_train, img_valid, mask_train, mask_valid = train_test_split(
        img_files, mask_files, test_size=split_ratio, random_state=42
    )

    # Copy files to train/valid directories
    for img_file, mask_file in zip(img_train, mask_train):
        shutil.copy(os.path.join(img_dir, img_file), train_img_dir)
        shutil.copy(os.path.join(mask_dir, mask_file), train_mask_dir)

    for img_file, mask_file in zip(img_valid, mask_valid):
        shutil.copy(os.path.join(img_dir, img_file), valid_img_dir)
        shutil.copy(os.path.join(mask_dir, mask_file), valid_mask_dir)

    print("bdappv dataset splitting completed!")


# Custom dataset class for semantic segmentation
class SemanticSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.transform = transform

        # Ensure that the number of images matches the number of masks
        if len(self.image_files) != len(self.mask_files):
            raise ValueError("The number of images and masks does not match!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
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

# Clean and split bdappv dataset
bdappv_data_dir = "/workspace/data/bdappv/google"
bdappv_img_dir = os.path.join(bdappv_data_dir, "img")
bdappv_mask_dir = os.path.join(bdappv_data_dir, "mask")
clean_bdappv_dataset(bdappv_img_dir, bdappv_mask_dir)
split_bdappv_dataset(bdappv_img_dir, bdappv_mask_dir, data_dir, split_ratio=0.3)

# Datasets for your main dataset
train_dataset_1 = SemanticSegmentationDataset(
    img_dir=train_data_dir,
    mask_dir=train_data_dir,
    transform=train_transform
)

valid_dataset_1 = SemanticSegmentationDataset(
    img_dir=valid_data_dir,
    mask_dir=valid_data_dir,
    transform=valid_transform
)

# Datasets for the bdappv dataset
train_dataset_2 = SemanticSegmentationDataset(
    img_dir=train_data_dir,
    mask_dir=train_data_dir,
    transform=train_transform
)

valid_dataset_2 = SemanticSegmentationDataset(
    img_dir=valid_data_dir,
    mask_dir=valid_data_dir,
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
num_epochs = 100
best_valid_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    valid_loss = validate_epoch(model, valid_loader, loss_fn, DEVICE)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")

