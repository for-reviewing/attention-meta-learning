# src/data/npz_to_torch.py
# This script needs at least around 72 GB RAM and 42.7 GB disk space

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# Path to the output PyTorch dataset file
pytorch_dataset_path = 'data/processed/pytorch_dataset.pt'

# Check if the PyTorch dataset file already exists
if os.path.exists(pytorch_dataset_path):
    print(f"PyTorch dataset already exists at {pytorch_dataset_path}. Skipping conversion.")
else:
    # Load images and labels from the NPZ file
    npz_file_path = 'data/interim/plant_village_data.npz'
    npz_file = np.load(npz_file_path)
    images = npz_file['images']
    labels = npz_file['labels']

    # Display the memory usage of the loaded data
    print(f"Loaded images and labels from NPZ file.")
    images_mem = images.nbytes
    labels_mem = labels.nbytes
    print(f"Memory usage: {(images_mem + labels_mem) / (1024 ** 3):.2f} GB")

    # Custom dataset class for PyTorch
    class CustomDataset(Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image, label = self.images[idx], self.labels[idx]
            if self.transform:
                image = self.transform(image)
            else:
                # Convert to tensor if no transform is specified
                image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            return image, label

    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the PyTorch dataset with transformations
    pytorch_dataset = CustomDataset(images, labels, transform=transform)

    # Save the PyTorch dataset for later use
    os.makedirs(os.path.dirname(pytorch_dataset_path), exist_ok=True)
    torch.save(pytorch_dataset, pytorch_dataset_path)
    print(f"PyTorch dataset saved as {pytorch_dataset_path}")