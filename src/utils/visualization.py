# src/utils/visualization.py

"""
Visualization Script for Feature Embeddings using t-SNE on PlantVillage Dataset with PyTorch

This script performs the following tasks:
1. Loads the trained models (with and without CBAM).
2. Loads a set of images from the dataset (e.g., unseen classes).
3. Extracts feature representations from the last layer before classification.
4. Applies t-SNE to reduce the feature dimensions for visualization.
5. Generates scatter plots to visualize how well the models cluster different classes.
6. Compares the feature embeddings of models with and without CBAM.

Requirements:
- Python 3.x
- Libraries: os, argparse, torch, torchvision, numpy, matplotlib, sklearn

Usage:
    python visualization.py --dataset_dir data/processed/dataset_MT10/unseen --model resnet18 --model_checkpoint models/checkpoints/maml_resnet18.pth --n_samples 1000 --output resnet18_tsne.png
    python visualization.py --dataset_dir data/processed/dataset_MT10/unseen --model resnet18_cbam --model_checkpoint models/checkpoints/maml_resnet18_cbam.pth --n_samples 1000 --output resnet18_cbam_tsne.png

Notes:
- Ensure that you have trained the models and have the model checkpoints available.
- Adjust hyperparameters and settings according to your needs.
"""

import os
import sys
sys.path.append(os.path.abspath(''))  # Ensure the root directory is in sys.path

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns  # Import seaborn for color palettes
import matplotlib.patheffects as pe  # Import patheffects for text effects


# Import the models defined in src.models.model_definition
from src.models.model_definition import ResNet18_CBAM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def extract_features(model, dataloader, device):
    """Extract features from the model for all images in the dataloader."""
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Extracting Features'):
            inputs = inputs.to(device)
            # Forward pass up to the last layer before classification
            output = model.features(inputs)
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def main():
    parser = argparse.ArgumentParser(description='Feature Visualization using t-SNE')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset (e.g., dataset_MT10/unseen)')
    parser.add_argument('--model', choices=['resnet18', 'resnet18_cbam'], required=True, help='Model to use')
    parser.add_argument('--model_checkpoint', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to use for visualization')
    parser.add_argument('--output', type=str, default='tsne_plot.png', help='Filename for the output plot')
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Include normalization consistent with training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    dataset = ImageFolder(root=args.dataset_dir, transform=transform)

    # Subsample the dataset if necessary
    if args.n_samples > 0 and args.n_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.n_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    dataloader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)

    # Define the model
    num_classes = len(dataset.classes)
    if args.model == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Identity()  # Remove the classification layer
    elif args.model == 'resnet18_cbam':
        model = ResNet18_CBAM(num_classes=num_classes)
        # Remove the classification layer
        model.model.fc = nn.Identity()
    else:
        raise ValueError('Invalid model selection')

    # Load the trained model checkpoint
    model_checkpoint_path = args.model_checkpoint  # Use the exact path provided

    # Load the state dict
    state_dict = torch.load(model_checkpoint_path)

    # Adjust state dict keys if necessary
    model_state_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # Check for 'model.' prefix mismatch
    if all(key.startswith('model.') for key in model_state_keys) and not any(key.startswith('model.') for key in checkpoint_keys):
        # Add 'model.' prefix to checkpoint keys
        new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
        state_dict = new_state_dict
    elif not any(key.startswith('model.') for key in model_state_keys) and all(key.startswith('model.') for key in checkpoint_keys):
        # Remove 'model.' prefix from checkpoint keys
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        state_dict = new_state_dict

    # Load the adjusted state dict into the model
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Modify the model to output features
    if args.model == 'resnet18':
        def features(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        model.features = features
    elif args.model == 'resnet18_cbam':
        def features(x):
            x = model.model.conv1(x)
            x = model.model.bn1(x)
            x = model.model.relu(x)
            x = model.model.maxpool(x)

            x = model.model.layer1(x)
            x = model.model.layer2(x)
            x = model.model.layer3(x)
            x = model.model.layer4(x)

            x = model.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        model.features = features

    # Extract features
    print('Extracting features...')
    features, labels = extract_features(model, dataloader, device)

    # Apply t-SNE
    print('Applying t-SNE...')
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plotting
    print('Plotting results...')
    palette = np.array(sns.color_palette("hls", num_classes))  # Choose color palette

    # Create a scatter plot.
    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot()
    sc = ax.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        lw=0,
        s=20,
        c=palette[labels.astype(int)],
        alpha=0.7
    )
    plt.grid()

    # Add the labels for each class at the median position
    # txts = []
    # for i in range(num_classes):
    #     # Position of each label.
    #     xtext, ytext = np.median(features_2d[labels == i, :], axis=0)
    #     txt = ax.text(
    #         xtext,
    #         ytext,
    #         dataset.classes[i],
    #         fontsize=12,
    #         path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()]
    #     )
    #     txts.append(txt)

    # Create custom legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            label=dataset.classes[i],
            markerfacecolor=palette[i],
            markersize=10
        ) for i in range(num_classes)
    ]
    ax.legend(handles=handles, loc='best')

    plt.title(f't-SNE Visualization of {args.model} Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.show()
    print(f'Plot saved as {args.output}')


if __name__ == '__main__':
    main()
