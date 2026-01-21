
# Smart Agriculture: Crop Pest & Disease Detection
ML system for detecting crop diseases using deep learning


[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning system for detecting crop diseases and pest infestations from leaf images. This project demonstrates a complete MLOps pipeline from data exploration through cloud deployment.

**Live Demo**: [Deployed on Fly.io](https://crop-disease-detection.fly.dev/docs)  
**ML Zoomcamp Capstone Project**: [DataTalks.Club](https://datatalks.club/)

---

## Table of Contents

- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [EDA](#eda)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

---

## Problem Description

Agricultural crop diseases and pest infestations cause significant yield losses worldwide, affecting:
- **Food Security**: Reduced crop yields impact global food supply
- **Economic Impact**: Farmers face substantial financial losses
- **Environmental Concerns**: Overuse of pesticides due to late detection
- **Knowledge Gap**: Manual inspection requires expert knowledge not always available

### The Solution

An automated deep learning system that:
- **Identifies 22 different crop diseases and pest conditions** from leaf images
- **Enables early detection** before visible symptoms spread
- **Provides instant diagnosis** via smartphone or web interface
- **Democratizes expertise** - accessible to farmers worldwide

### Use Cases

1. **Mobile App for Farmers**: Take a photo of affected crops and get instant diagnosis
2. **Automated Monitoring**: Deploy in fields with camera systems for continuous surveillance
3. **Agricultural Extension Services**: Support agents in providing better guidance
4. **Educational Tool**: Training resource for agricultural students

---

## Dataset

**Source**: [CCMT Dataset for Crop Pest and Disease Detection](https://www.kaggle.com/datasets/siddharthjiyani/ccmt-dataset-for-crop-pest-and-disease-detection) (Kaggle)

### Dataset Statistics

The CCMT Dataset is ideal, offering 24,881 raw images (6,549 cashew, 7,508 cassava, 5,389 maize, 5,435 tomato) and 102,976 augmented images across 22 disease classes. Validated by expert plant virologists, this dataset is tailored to Ghana’s agricultural context.
- **Format**: RGB images of crop leaves
- **Image Size**: Variable (resized to 224×224 for model input)

### Disease Classes

The model can identify the following conditions:

- Cashew__anthracnose
- Cashew__gumosis
- Cashew__healthy
- Cashew__leaf_miner
- Cashew__red_rust
- Cassava__bacterial_blight
- Cassava__brown_spot
- Cassava__green_mite
- Cassava__healthy
- Cassava__mosaic
- Corn_(maize)__fall_armyworm
- Corn_(maize)__grasshoper
- Corn_(maize)__healthy
- Corn_(maize)__leaf_beetle
- Corn_(maize)__leaf_blight
- Corn_(maize)__leaf_spot
- Corn_(maize)__streak_virus
- Tomato__healthy
- Tomato__leaf_blight
- Tomato__leaf_curl
- Tomato__septoria_leaf_spot
- Tomato__verticulium_wilt

---

## EDA
**Training Set Analysis**

**Class Balance Statistics:**
  - Total Images: 80271
  - Number of Classes: 22
  -  Samples per class - Min: 830, Max: 9373

**Validation Set Analysis**

**Class Balance Statistics:**
 -   Total Images: 24981
 -   Number of Classes: 22
 -   Samples per class - Min: 211, Max: 2623
<img width="1789" height="590" alt="2f83b33c-037a-4ef4-b1f9-14626830d018" src="https://github.com/user-attachments/assets/da3fbdd1-009b-48d3-b222-dd2e8793e2dc" />
<img width="897" height="4390" alt="5c1d5e00-902f-421b-973c-2498c715ff15" src="https://github.com/user-attachments/assets/bd87b04c-8155-4d7a-8d07-ccd564564f79" />



---

## Model Architecture

### Overview

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Framework**: PyTorch
- **Inference**: ONNX Runtime (optimized for production)
- **Input**: 224×224 RGB images
- **Output**: 22-class classification

### Architecture Details

```
Input (224×224×3)
    ↓
MobileNetV2 Features (pretrained)
    ↓
Global Average Pooling
    ↓
Dense Layer (1280 → 128)
    ↓
ReLU + Dropout (0.3)
    ↓
Output Layer (128 → 22)
```

### Training Configuration

```python
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss
Batch Size: 32
Epochs: 10
Data Augmentation:
  - Random Crop (scale: 0.8-1.0)
  - Random Horizontal/Vertical Flip
  - Random Rotation (±30°)
  - Color Jitter
Normalization: ImageNet stats
```

---

## Project Structure

```
crop-disease-detection/
├── app/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py               
│   └── utils.py               # Preprocessing utilities
├── src/                        # Training code
│   ├── __init__.py
│   ├── train.py               # Training script
│   ├── model.py               # Model architecture
│   └── dataset.py             # Dataset utilities
├── models/                     # Trained models
│   ├── crop_disease_model.onnx
│   └── classes.json
├── notebooks/                  # Jupyter notebooks
│   └── eda_and_training.ipynb
├── tests/                      # Unit tests
│   └── test_api.py
├── scripts/                    # Utility scripts
│   └── test_api.sh
├── Dockerfile                  # Docker configuration
├── fly.toml                   # Fly.io configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for containerization)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Nonso-Analytics/crop-disease-detection.git
cd crop-disease-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

```

### Run Locally

```bash
# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Visit http://localhost:8000/docs for interactive API documentation
```

---

## Local Development

### Training the Model

```bash
# Train with default parameters
python src/train.py

# Train with custom parameters
python src/train.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0005 \
    --dropout 0.5
```

### Testing the API

```bash
# Run unit tests
pytest tests/ -v

# Test with curl
curl -X POST http://localhost:8000/predict \
  -F "file=@0cashew_valid_anthracnose.jpg"

```

---

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t crop-disease-api:latest .
```

### Run Docker Container

```bash
# Run container
docker run -d \
  --name crop-disease-api \
  -p 8000:8000 \
  crop-disease-api:latest

```

---

## Cloud Deployment

### Deployed on Fly.io

The application is deployed on Fly.io for production use.

** Live API**: `[https://crop-disease-detection.fly.dev/docs](https://crop-disease-detection.fly.dev/docs)`

#### Deploy to Fly.io

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

flyctl auth login
flyctl launch
flyctl deploy

flyctl status
flyctl logs
```

---

## API Documentation

### Base URL

- **Local**: `http://localhost:8000`
- **Production**: `[[https://YOUR-APP-NAME.fly.dev](https://crop-disease-detection.fly.dev/docs)](https://crop-disease-detection.fly.dev/docs)`

### Endpoints

#### 1. Root Endpoint

```bash
GET /
```

Returns API information and available endpoints.

**Response**:
```json
{
  "message": "Crop Disease Detection API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "/predict",
    "health": "/health",
    "docs": "/docs"
  }
}
```

#### 2. Health Check

```bash
GET /health
```

Check API health and model status.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes_loaded": true,
  "num_classes": 22
}
```

#### 3. Predict Disease

```bash
POST /predict
```

Upload an image to get disease prediction.

**Request**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@leaf_image.jpg"
```

**Response**:
```json
{
  "prediction": "Early_Blight",
  "confidence": 0.94,
  "top_3_predictions": [
    {"class": "Early_Blight", "confidence": 0.94},
    {"class": "Late_Blight", "confidence": 0.03},
    {"class": "Septoria_Leaf_Spot", "confidence": 0.02}
  ],
  "inference_time_ms": 45.23
}
```

#### 4. Batch Prediction

```bash
POST /predict_batch
```

Predict multiple images at once (max 10).

**Request**:
```bash
curl -X POST http://localhost:8000/predict_batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

#### 5. Get All Classes

```bash
GET /classes
```

Get list of all disease classes.

**Response**:
```json
{
  "num_classes": 22,
  "classes": ["Anthracnose", "Bacterial_Blight", ...]
}
```

### Interactive Documentation

Visit `/docs` for Swagger UI or `/redoc` for ReDoc documentation.

---

## Model Performance


Per-class Accuracy:
  Cashew__anthracnose           : 0.7084 (1302/1838)
  Cashew__gumosis               : 0.9835 (418/425)
  Cashew__healthy               : 0.9499 (1269/1336)
  Cashew__leaf_miner            : 0.7270 (1081/1487)
  Cashew__red_rust              : 0.9554 (1734/1815)
  Cassava__bacterial_blight     : 0.8898 (2334/2623)
  Cassava__brown_spot           : 0.6271 (930/1483)
  Cassava__green_mite           : 0.7333 (748/1020)
  Cassava__healthy              : 0.8319 (985/1184)
  Cassava__mosaic               : 0.7117 (854/1200)
  Corn_(maize)__fall_armyworm   : 0.6303 (179/284)
  Corn_(maize)__grasshoper      : 0.7956 (327/411)
  Corn_(maize)__healthy         : 0.5877 (124/211)
  Corn_(maize)__leaf_beetle     : 0.8789 (835/950)
  Corn_(maize)__leaf_blight     : 0.7709 (774/1004)
  Corn_(maize)__leaf_spot       : 0.3973 (501/1261)
  Corn_(maize)__streak_virus    : 0.6614 (664/1004)
  Tomato__healthy               : 0.6940 (347/500)
  Tomato__leaf_blight           : 0.3407 (446/1309)
  Tomato__leaf_curl             : 0.2444 (130/532)
  Tomato__septoria_leaf_spot    : 0.8846 (2070/2340)
  Tomato__verticulium_wilt      : 0.0864 (66/764)

### Training Results

```

Starting training for 10 epochs...
Batch size: 32
Learning rate: 0.001
Device: cuda

Epoch 1/10
  Train Loss: 1.3460, Train Acc: 0.5549
  Val Loss: 0.9025, Val Acc: 0.6755
Checkpoint saved: crop_disease_model_epoch01_acc0.676.pth
Epoch 2/10
  Train Loss: 1.1312, Train Acc: 0.6134
  Val Loss: 0.8845, Val Acc: 0.6806
Checkpoint saved: crop_disease_model_epoch02_acc0.681.pth
Epoch 3/10
  Train Loss: 1.0881, Train Acc: 0.6277
  Val Loss: 0.8383, Val Acc: 0.6970
Checkpoint saved: crop_disease_model_epoch03_acc0.697.pth
Epoch 4/10
  Train Loss: 1.0698, Train Acc: 0.6325
  Val Loss: 0.8263, Val Acc: 0.7012
Checkpoint saved: crop_disease_model_epoch04_acc0.701.pth
Epoch 5/10
  Train Loss: 1.0524, Train Acc: 0.6364
  Val Loss: 0.8281, Val Acc: 0.7035
...
  Train Loss: 1.0082, Train Acc: 0.6516
  Val Loss: 0.7465, Val Acc: 0.7253

 Best validation accuracy: 0.7258
```
<img width="1872" height="1789" alt="83030613-4ec8-4de0-95bd-458c00e5d4b2" src="https://github.com/user-attachments/assets/842f8376-55d9-4dec-866e-35b09f85c465" />



### Visualizations

---

## Technologies Used

### Machine Learning
- **PyTorch** 2.1.1 - Deep learning framework
- **torchvision** 0.16.1 - Computer vision models
- **ONNX Runtime** 1.16.3 - Optimized inference
- **scikit-learn** 1.3.2 - Metrics and evaluation

### Web Framework
- **FastAPI** 0.104.1 - Modern web framework
- **Uvicorn** 0.24.0 - ASGI server
- **Python-multipart** - File upload support

### Data Processing
- **NumPy** 1.24.3 - Numerical computing
- **Pillow** 10.1.0 - Image processing
- **Matplotlib** 3.8.2 - Visualization
- **Seaborn** 0.13.0 - Statistical plots

### DevOps
- **Docker** - Containerization
- **Fly.io** - Cloud platfor

---

## Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test prediction with sample image
curl -X POST http://localhost:8000/predict \
  -F "file=@data/valid/Early_Blight/sample.jpg"

# Test with Python
python tests/test_manual.py

<img width="932" height="467" alt="image" src="https://github.com/user-attachments/assets/3d4a11ba-ff36-43ef-9003-70ec5ea8c4dc" />

