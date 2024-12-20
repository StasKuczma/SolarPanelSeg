import os

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp

import cv2

from dataset import SegmentationDataset

data_dir = "/workspace/data"

train_data_dir = os.path.join(data_dir, "train")
test_data_dir = os.path.join(data_dir, "test")

train_dataset = SegmentationDataset(data_dir=train_data_dir)


print(f"Number of samples in the dataset: {len(train_dataset)}")

sample=train_dataset[0]

plt.imshow(sample[0])
plt.axis("off") 
plt.savefig("output_image.png")