# pretrain_model.py

"""
Pretraining Script for ResNet18 with and without CBAM on ImageNet using PyTorch

This script performs the following tasks:
1. Loads the ImageNet dataset.
2. Defines the ResNet18 models (with and without CBAM).
3. Trains both models to learn general image features.
4. Saves the pretrained weights for initialization in meta-learning.

Requirements:
- Python 3.x
- Libraries: os, torch, torchvision, tqdm

Usage:
    python pretrain_model.py --data_dir /path/to/imagenet --epochs 90 --batch_size 256

Note:
- Training on the full ImageNet dataset requires significant computational resources.
- Adjust batch size and number of workers according to your hardware capabilities.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the models defined in model_definition.py
from src.models.model_definition import ResNet18_CBAM

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, model_name):
    """Function to train the model."""
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            loop = tqdm(dataloaders[phase], desc=f'{phase} Phase', leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f'{model_name}_epoch_{epoch + 1}.pth')

    # Save the final model
    torch.save(model.state_dict(), f'{model_name}_final.pth')
    print(f'Model {model_name} saved as {model_name}_final.pth')

def main():
    parser = argparse.ArgumentParser(description='Pretrain ResNet18 models on ImageNet')
    parser.add_argument('--data_dir', required=True, help='Path to ImageNet dataset')
    parser.add_argument('--epochs', type=int, default=90, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Create training and validation datasets
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms['val']),
    }

    # Create training and validation dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size,
                          shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }

    num_classes = 1000  # Number of classes in ImageNet

    # Define the models
    # Model without CBAM
    model_resnet18 = models.resnet18(pretrained=False)
    model_resnet18.fc = nn.Linear(512, num_classes)
    model_resnet18 = model_resnet18.to(device)

    # Model with CBAM
    model_resnet18_cbam = ResNet18_CBAM(num_classes=num_classes)
    model_resnet18_cbam = model_resnet18_cbam.to(device)

    # Define the loss function and optimizers
    criterion = nn.CrossEntropyLoss()

    # Optimizers for both models
    optimizer_resnet18 = optim.SGD(model_resnet18.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    optimizer_resnet18_cbam = optim.SGD(model_resnet18_cbam.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Learning rate schedulers
    scheduler_resnet18 = optim.lr_scheduler.StepLR(optimizer_resnet18, step_size=30, gamma=0.1)
    scheduler_resnet18_cbam = optim.lr_scheduler.StepLR(optimizer_resnet18_cbam, step_size=30, gamma=0.1)

    # Train the models
    print('Training ResNet18 without CBAM...')
    train_model(model_resnet18, dataloaders, criterion, optimizer_resnet18, scheduler_resnet18,
                num_epochs=args.epochs, device=device, model_name='resnet18')

    print('\nTraining ResNet18 with CBAM...')
    train_model(model_resnet18_cbam, dataloaders, criterion, optimizer_resnet18_cbam, scheduler_resnet18_cbam,
                num_epochs=args.epochs, device=device, model_name='resnet18_cbam')

if __name__ == '__main__':
    main()
