import os

from torch.utils.data import DataLoader, Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [
            f for f in os.listdir(data_dir) if not f.endswith("_mask.png")
        ] 
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace(".jpg", "_mask.png") 

        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
