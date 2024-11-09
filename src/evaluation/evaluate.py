# src/evaluation/evaluate.py

"""
Evaluation Script for Trained MAML Models on PlantVillage Dataset using PyTorch

This script performs the following tasks:
1. Loads the trained model from meta_training.py.
2. Loads the unseen data (unseen tasks) from the preprocessed PlantVillage dataset.
3. Generates tasks (episodes) for evaluation.
4. Uses the trained model to make predictions on the query sets of these tasks.
5. Computes performance metrics: accuracy, precision, recall, specificity, and F1-score.
6. Reports results with 95% confidence intervals.

Requirements:
- Python 3.x
- Libraries: os, argparse, torch, torchvision, numpy, tqdm, sklearn

Usage:
python evaluate.py --dataset_dir /path/to/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 10 --k_shot 5 --num_tasks 600 --model_checkpoint maml_model.pth

Example:
python src/evaluation/evaluate.py --dataset_dir data/processed/dataset_MT10/unseen --experiment MT10 --model resnet18_cbam --n_way 5 --k_shot 5 --q_queries 15 --num_tasks 600 --inner_steps 5 --inner_lr 0.01 --model_checkpoint models/checkpoints/maml_model.pth

Notes:
- Ensure that you have trained the model using meta_training.py and have the model checkpoint available.
- Adjust hyperparameters and settings according to your needs.
"""

import os
import sys
sys.path.append(os.path.abspath(''))  # Ensure the root directory is in sys.path

import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
from copy import deepcopy  # Import deepcopy

# Import the models defined in src.models.model_definition
from src.models.model_definition import ResNet18_CBAM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class FewShotDataset(Dataset):
    """
    Dataset class to generate tasks for N-way K-shot evaluation.
    """
    def __init__(self, data_root, n_way, k_shot, q_queries, transform=None):
        self.data_root = data_root
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.transform = transform

        # Get list of classes
        self.classes = sorted(os.listdir(data_root))  # Ensure consistent ordering
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Load all images paths for each class
        self.images_per_class = {}
        for cls in self.classes:
            cls_dir = os.path.join(data_root, cls)
            images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir)]
            self.images_per_class[cls] = images

    def __len__(self):
        # Return a large number since we'll sample tasks on the fly
        return 100000

    def __getitem__(self, idx):
        # Sample n_way classes
        sampled_classes = random.sample(self.classes, self.n_way)
        support_set = []
        query_set = []
        support_labels = []
        query_labels = []

        for i, cls in enumerate(sampled_classes):
            images = self.images_per_class[cls]
            if len(images) < self.k_shot + self.q_queries:
                # Not enough images to sample, so replicate images
                images = images * ((self.k_shot + self.q_queries) // len(images) + 1)
            sampled_images = random.sample(images, self.k_shot + self.q_queries)
            support_imgs = sampled_images[:self.k_shot]
            query_imgs = sampled_images[self.k_shot:]

            # Load and transform images
            for img_path in support_imgs:
                image = self.load_image(img_path)
                support_set.append(image)
                support_labels.append(i)
            for img_path in query_imgs:
                image = self.load_image(img_path)
                query_set.append(image)
                query_labels.append(i)

        # Convert lists to tensors
        support_set = torch.stack(support_set)
        query_set = torch.stack(query_set)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        return support_set, support_labels, query_set, query_labels

    def load_image(self, img_path):
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def compute_confidence_interval(data):
    """Compute 95% confidence interval for a list of accuracies."""
    mean = np.mean(data)
    std_error = stats.sem(data)
    h = std_error * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
    return mean, h

def compute_specificity(cm):
    """Compute specificity for each class and return the average."""
    num_classes = cm.shape[0]
    specificity_per_class = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP + 1e-8)
        specificity_per_class.append(specificity)
    return np.mean(specificity_per_class)

def main():
    parser = argparse.ArgumentParser(description='Evaluation of Trained MAML Models on PlantVillage')
    parser.add_argument('--dataset_dir', required=True, help='Path to unseen dataset (e.g., dataset_MT10/unseen)')
    parser.add_argument('--experiment', choices=['MT10', 'MT6'], required=True, help='Experiment setup (MT10 or MT6)')
    parser.add_argument('--model', choices=['resnet18', 'resnet18_cbam'], required=True, help='Model to use')
    parser.add_argument('--n_way', type=int, default=10, help='Number of classes per task (N-way)')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of samples per class in support set (K-shot)')
    parser.add_argument('--q_queries', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--num_tasks', type=int, default=600, help='Number of tasks for evaluation')
    parser.add_argument('--model_checkpoint', required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Inner loop learning rate during adaptation')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of inner loop updates during adaptation')
    args = parser.parse_args()

    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Include any normalization used during training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Update as per your training script
                             std=[0.229, 0.224, 0.225]),
    ])

    # Create the dataset
    dataset = FewShotDataset(data_root=args.dataset_dir,
                             n_way=args.n_way,
                             k_shot=args.k_shot,
                             q_queries=args.q_queries,
                             transform=transform)

    # Define the model
    num_classes = args.n_way  # For each task, the number of classes is N-way

    if args.model == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)
    elif args.model == 'resnet18_cbam':
        model = ResNet18_CBAM(num_classes=num_classes)
    else:
        raise ValueError('Invalid model selection')
    
    # After defining the model
    print("Model's state_dict keys:")
    for k in model.state_dict().keys():
        print(k)

    # Load the trained model checkpoint
    model_checkpoint_path = args.model_checkpoint  # Use the exact path provided

    # Load the state dict
    state_dict = torch.load(model_checkpoint_path)

    print("\nLoaded state_dict keys:")
    for k in state_dict.keys():
        print(k)

    # Load the trained model checkpoint
    model_checkpoint_path = args.model_checkpoint  # Use the exact path provided
    
    model.load_state_dict(torch.load(model_checkpoint_path))
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Metrics lists
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []

    print('Starting Evaluation...')
    for task_idx in tqdm(range(args.num_tasks), desc='Evaluating Tasks'):
        # Get a task
        support_set, support_labels, query_set, query_labels = dataset[task_idx]
        support_set = support_set.to(device)
        support_labels = support_labels.to(device)
        query_set = query_set.to(device)
        query_labels = query_labels.to(device)

        # Create a copy of the model for adaptation
        fast_model = deepcopy(model)
        fast_model.train()

        # Inner loop adaptation on support set
        fast_optimizer = torch.optim.SGD(fast_model.parameters(), lr=args.inner_lr)
        for _ in range(args.inner_steps):
            # Forward pass on support set
            support_outputs = fast_model(support_set)
            support_loss = criterion(support_outputs, support_labels)
            # Zero gradients, backward pass, and update weights
            fast_optimizer.zero_grad()
            support_loss.backward()
            fast_optimizer.step()

        # Evaluation on query set
        fast_model.eval()
        with torch.no_grad():
            query_outputs = fast_model(query_set)
            _, query_preds = torch.max(query_outputs, 1)

        # Move tensors to CPU for metric computation
        query_labels_cpu = query_labels.cpu().numpy()
        query_preds_cpu = query_preds.cpu().numpy()

        # Compute metrics
        accuracy = accuracy_score(query_labels_cpu, query_preds_cpu)
        precision = precision_score(query_labels_cpu, query_preds_cpu, average='macro', zero_division=0)
        recall = recall_score(query_labels_cpu, query_preds_cpu, average='macro', zero_division=0)
        f1 = f1_score(query_labels_cpu, query_preds_cpu, average='macro', zero_division=0)

        # Compute specificity for each class and take average
        cm = confusion_matrix(query_labels_cpu, query_preds_cpu)
        specificity = compute_specificity(cm)

        # Append metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)

        # Clean up
        del fast_model, support_set, support_labels, query_set, query_labels
        torch.cuda.empty_cache()

    # Compute mean and confidence intervals
    acc_mean, acc_ci = compute_confidence_interval(accuracies)
    prec_mean, prec_ci = compute_confidence_interval(precisions)
    rec_mean, rec_ci = compute_confidence_interval(recalls)
    spec_mean, spec_ci = compute_confidence_interval(specificities)
    f1_mean, f1_ci = compute_confidence_interval(f1_scores)

    # Print results
    print('\nEvaluation Results:')
    print(f'Accuracy: {acc_mean*100:.2f}% ± {acc_ci*100:.2f}%')
    print(f'Precision: {prec_mean*100:.2f}% ± {prec_ci*100:.2f}%')
    print(f'Recall: {rec_mean*100:.2f}% ± {rec_ci*100:.2f}%')
    print(f'Specificity: {spec_mean*100:.2f}% ± {spec_ci*100:.2f}%')
    print(f'F1-Score: {f1_mean*100:.2f}% ± {f1_ci*100:.2f}%')

if __name__ == '__main__':
    main()
