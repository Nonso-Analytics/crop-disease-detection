"""
Dataset utilities for Crop Disease Detection
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CropDiseaseDataset(Dataset):
    """
    Custom Dataset for CCMT Crop Disease images
    
    Args:
        data_dir: Path to data directory containing class subdirectories
        transform: Optional transform to be applied on images
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get all disease classes (sorted for consistency)
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(label_dir, img_name))
                        self.labels.append(self.class_to_idx[label_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size=224):
    """
    Get training and validation transforms
    
    Args:
        input_size: Size of input images (default: 224)
    
    Returns:
        train_transform, val_transform
    """
    # ImageNet normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


def get_inference_transform(input_size=224):
    """
    Get transform for inference
    
    Args:
        input_size: Size of input images (default: 224)
    
    Returns:
        Inference transform
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])