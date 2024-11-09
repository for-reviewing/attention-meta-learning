# src/models/model_definition.py

"""
Model Definition Script for ResNet18 with and without CBAM using PyTorch

This script performs the following tasks:
1. Defines the ResNet18 architecture without CBAM.
2. Implements the Convolutional Block Attention Module (CBAM).
3. Integrates CBAM into the ResNet18 architecture.
4. Adds a method to support forward passes with custom weights (forward_with_weights).
5. Optionally loads pre-trained weights if 'pretrained' is set to True.

Requirements:
- Python 3.x
- Libraries: torch, torchvision

Usage:
    python model_definition.py --pretrained True
"""

import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import torch.nn.functional as F

# Define the CBAM module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention Module
        self.channel_attention = ChannelAttention(channels, reduction)
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Global Average Pooling and Global Max Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # The kernel size must be odd and greater than or equal to 3
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Concatenate along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

# Modify ResNet18 to include CBAM
class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18_CBAM, self).__init__()
        # Load the standard ResNet18 model
        self.model = models.resnet18(weights=None)
        # Modify the fully connected layer to match the number of classes
        self.model.fc = nn.Linear(512, num_classes)
        # Add CBAM after each residual block
        self._add_cbam()

    def _add_cbam(self):
        """Add CBAM to each residual block in the ResNet18 model."""
        def make_cbam_layer(layer):
            for idx in range(len(layer)):
                layer[idx].conv1 = nn.Sequential(
                    layer[idx].conv1,
                    CBAM(layer[idx].conv1.out_channels)
                )
            return layer
        
        self.model.layer1 = make_cbam_layer(self.model.layer1)
        self.model.layer2 = make_cbam_layer(self.model.layer2)
        self.model.layer3 = make_cbam_layer(self.model.layer3)
        self.model.layer4 = make_cbam_layer(self.model.layer4)
    
    def forward(self, x):
        return self.model(x)

    def forward_with_weights(self, x, weights):
        # Manually define the forward pass using the provided weights
        x = F.conv2d(x, weights['model.conv1.weight'], weights.get('model.conv1.bias'), stride=2, padding=3)
        x = F.batch_norm(x, running_mean=self.model.bn1.running_mean, running_var=self.model.bn1.running_var,
                         weight=weights['model.bn1.weight'], bias=weights['model.bn1.bias'], training=self.model.bn1.training)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self._forward_layer(x, self.model.layer1, weights, layer_name='model.layer1')
        x = self._forward_layer(x, self.model.layer2, weights, layer_name='model.layer2')
        x = self._forward_layer(x, self.model.layer3, weights, layer_name='model.layer3')
        x = self._forward_layer(x, self.model.layer4, weights, layer_name='model.layer4')

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, weights['model.fc.weight'], weights['model.fc.bias'])
        return x

    def _forward_layer(self, x, layer, weights, layer_name):
        for idx, block in enumerate(layer):
            residual = x

            # Conv1
            conv1_weight = weights[f'{layer_name}.{idx}.conv1.0.weight']
            x = F.conv2d(x, conv1_weight, stride=block.conv1[0].stride, padding=block.conv1[0].padding)
            # BatchNorm1
            bn1_weight = weights[f'{layer_name}.{idx}.bn1.weight']
            bn1_bias = weights[f'{layer_name}.{idx}.bn1.bias']
            x = F.batch_norm(x, running_mean=block.bn1.running_mean, running_var=block.bn1.running_var,
                            weight=bn1_weight, bias=bn1_bias, training=block.bn1.training)
            x = block.relu(x)

            # Conv2
            conv2_weight = weights[f'{layer_name}.{idx}.conv2.weight']
            x = F.conv2d(x, conv2_weight, stride=block.conv2.stride, padding=block.conv2.padding)
            # BatchNorm2
            bn2_weight = weights[f'{layer_name}.{idx}.bn2.weight']
            bn2_bias = weights[f'{layer_name}.{idx}.bn2.bias']
            x = F.batch_norm(x, running_mean=block.bn2.running_mean, running_var=block.bn2.running_var,
                            weight=bn2_weight, bias=bn2_bias, training=block.bn2.training)

            # Handle the downsample layer using custom weights
            if block.downsample is not None:
                # Downsample consists of Conv and BatchNorm
                # Conv layer
                ds_conv_weight = weights[f'{layer_name}.{idx}.downsample.0.weight']
                ds_conv_bias = weights.get(f'{layer_name}.{idx}.downsample.0.bias')
                residual = F.conv2d(residual, ds_conv_weight, bias=ds_conv_bias, stride=block.downsample[0].stride, padding=block.downsample[0].padding)
                # BatchNorm layer
                ds_bn_weight = weights[f'{layer_name}.{idx}.downsample.1.weight']
                ds_bn_bias = weights[f'{layer_name}.{idx}.downsample.1.bias']
                residual = F.batch_norm(residual, running_mean=block.downsample[1].running_mean, running_var=block.downsample[1].running_var,
                                        weight=ds_bn_weight, bias=ds_bn_bias, training=block.downsample[1].training)

            # Skip connection
            x += residual
            x = block.relu(x)
        return x

def save_model(model, model_name):
    """Save the PyTorch model's state_dict."""
    model_dir = 'models/checkpoints/'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))
    print(f'Model saved as {model_dir}{model_name}.pth')

def main():
    parser = argparse.ArgumentParser(description='Define and Save Models')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained weights')
    args = parser.parse_args()

    # Number of classes in your dataset
    num_classes = 38  # Adjust according to your dataset

    # Define the ResNet18 model without CBAM
    from torchvision.models import resnet18, ResNet18_Weights

    if args.pretrained:
        weights = ResNet18_Weights.DEFAULT
        model_resnet18 = resnet18(weights=weights)
    else:
        model_resnet18 = resnet18(weights=None)
    # Modify the final fully connected layer
    model_resnet18.fc = nn.Linear(512, num_classes)
    # Save the model
    save_model(model_resnet18, 'resnet18')

    # Define the ResNet18 model with CBAM
    model_resnet18_cbam = ResNet18_CBAM(num_classes=num_classes)

    if args.pretrained:
        # Load pre-trained weights into the backbone of ResNet18_CBAM
        pretrained_resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        pretrained_dict = pretrained_resnet18.state_dict()
        model_dict = model_resnet18_cbam.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the new state dict
        model_resnet18_cbam.load_state_dict(model_dict)
        print('Pre-trained weights loaded into ResNet18_CBAM backbone.')
    else:
        print('ResNet18_CBAM initialized with random weights.')

    # Save the model
    save_model(model_resnet18_cbam, 'resnet18_cbam')

if __name__ == '__main__':
    main()
