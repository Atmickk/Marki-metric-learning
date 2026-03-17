import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
from dataset import ArtistImageDataset
from torchvision import transforms
from config import TRAIN_CONFIG, DATALOADER_CONFIG
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration from config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_val.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "Other_Marks")

# Hyperparameters from config
M = DATALOADER_CONFIG['m_per_class']  # images per class per batch
BATCH_SIZE = TRAIN_CONFIG['batch_size']  # should be m * n_classes_per_batch
NUM_WORKERS = DATALOADER_CONFIG['num_workers']
IMAGE_SIZE = 224

# Simple transform (no augmentation) - for validation
simple_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Albumentations augmentation pipeline for training
_albu_train = A.Compose([
    #A.HorizontalFlip(p=0.5),
    #A.Rotate(limit=20, p=0.6),
    #A.Perspective(scale=(0.05, 0.1), p=0.3),
    #A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
    A.ToGray(p=0.1),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    #A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
    #A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 40), hole_width_range=(16, 40), fill=0, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Augmented transform for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: _albu_train(image=np.array(img))["image"]),
])

# Train Dataset (with augmentation)
train_dataset = ArtistImageDataset(TRAIN_CSV, IMAGE_DIR, transform=train_transform)
train_labels = train_dataset.labels

# Train Sampler
train_sampler = MPerClassSampler(train_labels, m=M, length_before_new_iter=len(train_dataset))

# Train DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=DATALOADER_CONFIG['pin_memory'] and torch.cuda.is_available(),
    drop_last=True
)

# Validation Dataset (no augmentation)
val_dataset = ArtistImageDataset(VAL_CSV, IMAGE_DIR, transform=simple_transform)
val_labels = val_dataset.labels

# Validation Sampler
val_sampler = MPerClassSampler(val_labels, m=M, length_before_new_iter=len(val_dataset))

# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=DATALOADER_CONFIG['pin_memory'] and torch.cuda.is_available(),
    drop_last=True
)

# Test Dataset (if available)
TEST_CSV = os.path.join(BASE_DIR, "labels", "final_labels2_test.csv")
test_loader = None

if os.path.exists(TEST_CSV):
    test_dataset = ArtistImageDataset(TEST_CSV, IMAGE_DIR, transform=simple_transform)
    test_labels = test_dataset.labels

    # For testing, we can use regular sequential loading
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DATALOADER_CONFIG['pin_memory'] and torch.cuda.is_available(),
        drop_last=False
    )
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
else:
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: Not found")

print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
