# src/data/data_preprocessing.py

"""
Data Preprocessing Script for PlantVillage Dataset using PyTorch

This script performs the following tasks:
1. Loads the PlantVillage dataset using ImageFolder.
2. Resizes images to 224x224 pixels.
3. Balances classes by augmenting underrepresented classes to 1,500 samples using torchvision.transforms.
4. Splits the dataset according to experimental setups (MT-10 and MT-6).
5. Saves the preprocessed images to designated directories for use with PyTorch's Dataset and DataLoader.

Requirements:
- Python 3.x
- Libraries: os, shutil, tqdm, torch, torchvision

Usage:
    python data_preprocessing.py
"""

import os
import shutil
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms, datasets

# Set the root directory of the PlantVillage dataset
DATASET_DIR = 'data/raw/plant_village/downloads/extracted/Plant_leave_diseases_dataset_without_augmentation'
PREPROCESSED_DIR = 'data/processed/preprocessed_plantvillage_dataset'

# Desired image size
IMAGE_SIZE = (224, 224)

# Number of samples per class after balancing
TARGET_SAMPLES_PER_CLASS = 1500

# Experimental setups
MT10_UNSEEN_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

MT6_UNSEEN_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___healthy'
]

def get_transforms(augment=False):
    """Get the image transformations."""
    if augment:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
    return transform

def balance_classes(dataset, class_to_idx):
    """Balance the classes by augmenting underrepresented classes."""
    # Count samples per class
    class_counts = {cls: 0 for cls in class_to_idx.keys()}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] += 1

    # Create a directory to save balanced dataset
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Process each class
    for class_name in tqdm(dataset.classes, desc='Balancing Classes'):
        class_idx = class_to_idx[class_name]
        class_samples = [s for s in dataset.samples if s[1] == class_idx]
        num_samples = len(class_samples)
        augmentation_needed = TARGET_SAMPLES_PER_CLASS - num_samples

        # Create output directory for the class
        class_output_dir = os.path.join(PREPROCESSED_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Transformations
        transform_basic = get_transforms(augment=False)
        transform_augment = get_transforms(augment=True)

        # Process original images
        for i, (img_path, _) in enumerate(class_samples):
            image = Image.open(img_path).convert('RGB')
            image = transform_basic(image)
            # Save the tensor as an image
            save_image_tensor(image, os.path.join(class_output_dir, f'{i}.png'))

        # Augment images if needed
        if augmentation_needed > 0:
            augment_times = augmentation_needed // num_samples + 1
            augmented_images = []
            for _ in range(augment_times):
                for img_path, _ in class_samples:
                    image = Image.open(img_path).convert('RGB')
                    image = transform_augment(image)
                    augmented_images.append(image)
                    if len(augmented_images) >= augmentation_needed:
                        break
                if len(augmented_images) >= augmentation_needed:
                    break

            # Save augmented images
            for i, image in enumerate(augmented_images):
                idx = num_samples + i
                save_image_tensor(image, os.path.join(class_output_dir, f'{idx}.png'))
                if idx + 1 >= TARGET_SAMPLES_PER_CLASS:
                    break

def save_image_tensor(tensor, filename):
    """Save a tensor as an image file."""
    from torchvision.utils import save_image
    save_image(tensor, filename)

def create_dataset_splits(classes, unseen_classes, output_base_dir):
    """Split the dataset into training and unseen sets according to experimental setups."""
    seen_classes = [cls for cls in classes if cls not in unseen_classes]

    # Create directories
    train_dir = os.path.join(output_base_dir, 'train')
    unseen_dir = os.path.join(output_base_dir, 'unseen')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(unseen_dir, exist_ok=True)

    # Move images to the appropriate directories
    for cls in seen_classes:
        src_dir = os.path.join(PREPROCESSED_DIR, cls)
        dst_dir = os.path.join(train_dir, cls)
        shutil.copytree(src_dir, dst_dir)

    for cls in unseen_classes:
        src_dir = os.path.join(PREPROCESSED_DIR, cls)
        dst_dir = os.path.join(unseen_dir, cls)
        shutil.copytree(src_dir, dst_dir)

def main():
    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=DATASET_DIR)

    # Balance classes and save preprocessed images
    print("Balancing classes and saving preprocessed images...")
    balance_classes(dataset, dataset.class_to_idx)

    # Create dataset splits for MT-10 experiment
    print("\nCreating dataset splits for MT-10 experiment...")
    mt10_output_dir = 'data/processed/dataset_MT10'
    create_dataset_splits(dataset.classes, MT10_UNSEEN_CLASSES, mt10_output_dir)

    # Create dataset splits for MT-6 experiment
    print("\nCreating dataset splits for MT-6 experiment...")
    mt6_output_dir = 'data/processed/dataset_MT6'
    create_dataset_splits(dataset.classes, MT6_UNSEEN_CLASSES, mt6_output_dir)

if __name__ == '__main__':
    main()
