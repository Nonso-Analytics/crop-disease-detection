"""
FastAPI application for Crop Disease Detection
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import onnxruntime as ort

from .utils import preprocess_image, load_class_names


# Initialize FastAPI app
app = FastAPI(
    title="Crop Disease Detection API",
    description="API for detecting crop diseases and pests from leaf images",
    version="1.0.0"
)

# Global variables for model and classes
MODEL_PATH = os.getenv("MODEL_PATH", "models/crop_disease_model.onnx")
CLASSES_PATH = os.getenv("CLASSES_PATH", "models/classes.json")
INPUT_SIZE = 224  # Changed to int for consistency

# Load model and classes at startup
ort_session = None
class_names = None


@app.on_event("startup")
async def load_model():
    """Load ONNX model and class names on startup"""
    global ort_session, class_names
    
    try:
        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        ort_session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Load class names
        class_names = load_class_names(CLASSES_PATH)
        print(f"✓ Loaded {len(class_names)} classes")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crop Disease Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = ort_session is not None
    classes_loaded = class_names is not None
    
    return {
        "status": "healthy" if (model_loaded and classes_loaded) else "unhealthy",
        "model_loaded": model_loaded,
        "classes_loaded": classes_loaded,
        "num_classes": len(class_names) if classes_loaded else 0
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict crop disease from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with prediction results
    """
    if ort_session is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and preprocess image
        start_time = time.time()
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess for model
        input_tensor = preprocess_image(image, INPUT_SIZE)
        
        # Get input name dynamically
        input_name = ort_session.get_inputs()[0].name
        
        # Run inference
        outputs = ort_session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]  # Shape: (num_classes,)
        
        # Get probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        # Get top prediction
        top_idx = np.argmax(probabilities)
        prediction = class_names[top_idx]
        confidence = float(probabilities[top_idx])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                "class": class_names[idx],
                "confidence": float(probabilities[idx])
            }
            for idx in top_3_indices
        ]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        # More detailed error logging
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict crop diseases from multiple uploaded images
    
    Args:
        files: List of uploaded image files
    
    Returns:
        JSON with batch prediction results
    """
    if ort_session is None or class_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch"
        )
    
    results = []
    start_time = time.time()
    
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "Not an image file"
            })
            continue
        
        try:
            # Read and preprocess image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = preprocess_image(image, INPUT_SIZE)
            
            # Run inference
            outputs = ort_session.run(None, {'input': input_tensor})
            logits = outputs[0][0]
            
            # Get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            
            # Get top prediction
            top_idx = np.argmax(probabilities)
            
            results.append({
                "filename": file.filename,
                "prediction": class_names[top_idx],
                "confidence": float(probabilities[top_idx])
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    total_time = (time.time() - start_time) * 1000
    
    return {
        "num_images": len(files),
        "results": results,
        "total_time_ms": round(total_time, 2)
    }


@app.get("/classes")
async def get_classes():
    """Get list of all disease classes"""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Classes not loaded")
    
    return {
        "num_classes": len(class_names),
        "classes": class_names
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)