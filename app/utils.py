"""
Utility functions for FastAPI application
"""

import json
import numpy as np
from PIL import Image
from typing import List, Union, Tuple


def preprocess_image(image: Image.Image, input_size: Union[int, Tuple[int, int]] = 224) -> np.ndarray:
    """
    Preprocess image for ONNX model inference
    
    Args:
        image: PIL Image in RGB format
        input_size: Target size (int or tuple)
    
    Returns:
        Preprocessed image as numpy array with shape (1, 3, H, W)
    """
    # Handle tuple or int input
    if isinstance(input_size, int):
        target_size = (input_size, input_size)
    else:
        target_size = input_size
    
    # Resize image
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_array = (img_array - mean) / std
    
    # Transpose from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension: CHW -> NCHW
    img_array = np.expand_dims(img_array, axis=0)
    
    # Ensure float32 dtype
    img_array = img_array.astype(np.float32)
    
    return img_array


def load_class_names(classes_path: str) -> List[str]:
    """
    Load class names from JSON file
    
    Supports two formats:
    1. Direct list: ["class1", "class2", ...]
    2. Dictionary: {"classes": ["class1", "class2", ...], "class_to_idx": {...}}
    
    Args:
        classes_path: Path to classes.json file
    
    Returns:
        List of class names
    """
    try:
        with open(classes_path, 'r') as f:
            class_info = json.load(f)
        
        # Handle list format
        if isinstance(class_info, list):
            return class_info
        
        # Handle dictionary format
        elif isinstance(class_info, dict):
            if 'classes' in class_info:
                return class_info['classes']
            else:
                raise ValueError("Dictionary must contain 'classes' key")
        
        else:
            raise ValueError(f"Unsupported format: {type(class_info)}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in classes file: {e}")


def postprocess_predictions(
    logits: np.ndarray,
    class_names: List[str],
    top_k: int = 3
) -> dict:
    """
    Post-process model predictions with softmax and top-k
    
    Args:
        logits: Raw model outputs (shape: (num_classes,))
        class_names: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    # Get top-k predictions
    top_k = min(top_k, len(probabilities))
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