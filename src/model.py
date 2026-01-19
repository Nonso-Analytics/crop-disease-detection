"""
Model architecture for Crop Disease Classification
"""

import torch
import torch.nn as nn
from torchvision import models


class CropDiseaseClassifier(nn.Module):
    """
    Crop Disease Classifier based on MobileNetV2
    
    Args:
        num_classes: Number of disease classes to predict
        size_inner: Size of the inner dense layer
        droprate: Dropout rate for regularization
    """
    
    def __init__(self, num_classes=22, size_inner=128, droprate=0.3):
        super(CropDiseaseClassifier, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze base model parameters for transfer learning
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Remove original classifier
        self.base_model.classifier = nn.Identity()
        
        # Custom classification head
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
        self.output_layer = nn.Linear(size_inner, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    def unfreeze_base(self):
        """Unfreeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)