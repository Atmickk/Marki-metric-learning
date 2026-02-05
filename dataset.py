import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings

class ArtistImageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, image_size=128):
        # Validate inputs
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_size = image_size

        # Default transform with configurable image size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Validate required columns
        if 'filename' not in self.data.columns or 'artist' not in self.data.columns:
            raise ValueError("CSV must contain 'filename' and 'artist' columns")

        self.filenames = self.data['filename'].tolist()
        self.labels = self.data['artist'].astype('category').cat.codes.tolist()
        self.artist_to_label = dict(zip(self.data['artist'], self.data['artist'].astype('category').cat.codes))

        # Validate image files exist
        self._validate_images()

    def _validate_images(self):
        """Check if image files exist and warn about missing ones"""
        missing = []
        for filename in self.filenames:
            img_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(img_path):
                missing.append(filename)

        if missing:
            warnings.warn(f"Found {len(missing)} missing images out of {len(self.filenames)}")
            if len(missing) <= 10:
                for fname in missing:
                    warnings.warn(f"  Missing: {fname}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])

        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # If image loading fails, return a black image placeholder
            warnings.warn(f"Error loading image {self.filenames[idx]}: {str(e)}")
            image = torch.zeros((3, self.image_size, self.image_size))

        label = self.labels[idx]
        return image, label

# Example usage:
# train_csv = "labels/final_labels2_train.csv"
# image_dir = "Other_Marks"
# dataset = ArtistImageDataset(train_csv, image_dir)
# img, label = dataset[0]
