"""
Utility functions for FastAPI application
"""

import json
import numpy as np
from PIL import Image
from typing import List


def preprocess_image(image: Image.Image, input_size: int = 224) -> np.ndarray:
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image
        input_size: Target size for the image
    
    Returns:
        Preprocessed image as numpy array
    """
    # Resize image
    image = image.resize((input_size, input_size))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Transpose to (C, H, W) format
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def load_class_names(classes_path: str) -> List[str]:
    """
    Load class names from JSON file
    
    Args:
        classes_path: Path to classes.json file
    
    Returns:
        List of class names
    """
    with open(classes_path, 'r') as f:
        class_info = json.load(f)
    return class_info['classes']


def postprocess_predictions(logits: np.ndarray, class_names: List[str], top_k: int = 3):
    """
    Post-process model predictions
    
    Args:
        logits: Raw model outputs
        class_names: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get top-k predictions
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    predictions = [
        {
            'class': class_names[idx],
            'confidence': float(probabilities[idx]),
            'class_index': int(idx)
        }
        for idx in top_indices
    ]
    
    return {
        'top_prediction': predictions[0],
        'top_k_predictions': predictions,
        'all_probabilities': probabilities.tolist()
    }