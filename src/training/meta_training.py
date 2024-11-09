# src/training/meta_training.py

"""
Meta-Learning Training Script using MAML with the 'higher' library on PlantVillage Dataset with PyTorch

This script performs the following tasks:
1. Loads the preprocessed PlantVillage dataset using ImageFolder.
2. Defines the ResNet18 models (with and without CBAM).
3. Implements the Model-Agnostic Meta-Learning (MAML) algorithm using the 'higher' library.
4. Trains the models on few-shot learning tasks generated from the dataset.
5. Optionally loads pre-trained weights if 'pretrained' is set to True.
6. Saves the trained models with filenames that include hyperparameters for easy identification.

Requirements:
- Python 3.x
- Libraries: os, argparse, torch, torchvision, numpy, tqdm, higher

Usage:
python src/training/meta_training.py --dataset_dir data/processed/dataset_MT10/train --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 5 --q_queries 15 --meta_batch_size 2 --inner_steps 5 --pretrained

Notes:
- Ensure that you have preprocessed the PlantVillage dataset using data_preprocessing.py.
- Adjust hyperparameters and settings according to your computational resources.
"""

import os
import sys
sys.path.append(os.path.abspath(''))  # Ensure the root directory is in sys.path

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random

# Import the models defined in src.models.model_definition
from src.models.model_definition import ResNet18_CBAM

# Import higher for meta-learning
import higher

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class FewShotDataset(Dataset):
    """
    Dataset class to generate tasks for N-way K-shot learning using ImageFolder.
    """
    def __init__(self, data_root, n_way, k_shot, q_queries, transform=None):
        self.data_root = data_root
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.transform = transform

        # Use ImageFolder to load the dataset
        self.dataset = datasets.ImageFolder(root=data_root)
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())

        # Organize images by class index
        self.images_per_class = {class_idx: [] for class_idx in self.class_to_idx.values()}
        for img_path, label in self.dataset.samples:
            self.images_per_class[label].append(img_path)

    def __len__(self):
        # Return a large number since we'll sample tasks on the fly
        return 100000

    def __getitem__(self, idx):
        # Sample n_way classes
        sampled_class_indices = random.sample(list(self.images_per_class.keys()), self.n_way)
        support_set = []
        query_set = []
        support_labels = []
        query_labels = []

        for i, class_idx in enumerate(sampled_class_indices):
            images = self.images_per_class[class_idx]
            # Ensure enough images are available
            if len(images) < self.k_shot + self.q_queries:
                images = images * ((self.k_shot + self.q_queries) // len(images) + 1)
            sampled_images = random.sample(images, self.k_shot + self.q_queries)
            support_imgs = sampled_images[:self.k_shot]
            query_imgs = sampled_images[self.k_shot:]

            # Load and transform images
            for img_path in support_imgs:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                support_set.append(image)
                support_labels.append(i)
            for img_path in query_imgs:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                query_set.append(image)
                query_labels.append(i)

        # Convert lists to tensors
        support_set = torch.stack(support_set)
        query_set = torch.stack(query_set)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_set, support_labels, query_set, query_labels

def main():
    parser = argparse.ArgumentParser(description='Meta-Learning Training with MAML on PlantVillage using higher')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset (e.g., data/processed/dataset_MT10/train)')
    parser.add_argument('--experiment', choices=['MT10', 'MT6'], required=True, help='Experiment setup (MT10 or MT6)')
    parser.add_argument('--model', choices=['resnet18', 'resnet18_cbam'], required=True, help='Model to use')
    parser.add_argument('--n_way', type=int, default=10, help='Number of classes per task (N-way)')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of samples per class in support set (K-shot)')
    parser.add_argument('--q_queries', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--meta_batch_size', type=int, default=2, help='Number of tasks per meta-update')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate')
    parser.add_argument('--num_iterations', type=int, default=5000, help='Number of meta-training iterations')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of inner loop updates')
    parser.add_argument('--save_model', type=str, default='', help='Filename to save the trained model')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained weights')
    parser.add_argument('--pretrained_model_path', type=str, default='', help='Path to pre-trained model weights')
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transformations consistent with data_preprocessing.py
    transform = transforms.Compose([
        # Apply the same augmentations used during preprocessing
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Create the dataset
    dataset = FewShotDataset(
        data_root=args.dataset_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        transform=transform
    )

    # Define the model
    num_classes = args.n_way  # For each task, the number of classes is N-way

    if args.model == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights

        if args.pretrained:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        else:
            model = resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)
    elif args.model == 'resnet18_cbam':
        model = ResNet18_CBAM(num_classes=num_classes)
        if args.pretrained:
            if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
                # Load pre-trained weights from the specified path
                model.load_state_dict(torch.load(args.pretrained_model_path))
                print('Pre-trained weights loaded from specified path into ResNet18_CBAM.')
            else:
                # Load pre-trained weights into the backbone of ResNet18_CBAM
                pretrained_resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                pretrained_dict = pretrained_resnet18.state_dict()
                model_dict = model.state_dict()
                # Filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                   if k in model_dict and v.size() == model_dict[k].size()}
                # Overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # Load the new state dict
                model.load_state_dict(model_dict)
                print('Pre-trained weights loaded into ResNet18_CBAM backbone.')
        else:
            print('ResNet18_CBAM initialized with random weights.')
    else:
        raise ValueError('Invalid model selection')

    model = model.to(device)

    # Define the meta-optimizer
    meta_optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print('Starting Meta-Training...')
    for iteration in range(1, args.num_iterations + 1):
        meta_optimizer.zero_grad()
        meta_loss = 0.0
        total_query_accuracy = 0.0

        # Sample a batch of tasks
        for task_idx in range(args.meta_batch_size):
            # Get a task
            support_set, support_labels, query_set, query_labels = dataset[iteration * args.meta_batch_size + task_idx]
            support_set = support_set.to(device)
            support_labels = support_labels.to(device)
            query_set = query_set.to(device)
            query_labels = query_labels.to(device)

            # Inner loop optimizer
            inner_optimizer = torch.optim.SGD(model.parameters(), lr=args.inner_lr)

            # Use higher to create a "monkey-patched" version of the model
            with higher.innerloop_ctx(
                model,
                inner_optimizer,
                copy_initial_weights=False,
                track_higher_grads=False
            ) as (fast_model, diff_optim):
                for _ in range(args.inner_steps):
                    support_outputs = fast_model(support_set)
                    support_loss = criterion(support_outputs, support_labels)
                    diff_optim.step(support_loss)

                # Compute loss on query set with the adapted model
                query_outputs = fast_model(query_set)
                query_loss = criterion(query_outputs, query_labels)
                # Accumulate meta-loss
                meta_loss += query_loss

                # Compute accuracy
                with torch.no_grad():
                    _, predicted = torch.max(query_outputs, 1)
                    correct = (predicted == query_labels).sum().item()
                    total = query_labels.size(0)
                    query_accuracy = correct / total
                    total_query_accuracy += query_accuracy

            # Clear variables to save memory
            del support_set, support_labels, query_set, query_labels
            torch.cuda.empty_cache()

        # Compute meta-gradient and update meta-parameters
        meta_loss /= args.meta_batch_size
        meta_loss.backward()
        meta_optimizer.step()

        # Average query accuracy
        avg_query_accuracy = total_query_accuracy / args.meta_batch_size

        # Logging
        if iteration % 100 == 0:
            print(f'Iteration {iteration}/{args.num_iterations}, Meta Loss: {meta_loss.item():.4f}, Query Accuracy: {avg_query_accuracy:.4f}')

    # Create a filename that includes hyperparameters
    if args.save_model:
        model_filename = args.save_model
    else:
        model_filename = f"maml_{args.model}_nway{args.n_way}_kshot{args.k_shot}_innerlr{args.inner_lr}_innersteps{args.inner_steps}_metalr{args.meta_lr}_metabatch{args.meta_batch_size}_iter{args.num_iterations}.pth"

    # Save the trained model
    save_model_path = os.path.join('models', 'checkpoints', model_filename)
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(model.state_dict(), save_model_path)
    print(f'Model saved as {save_model_path}')

if __name__ == '__main__':
    main()