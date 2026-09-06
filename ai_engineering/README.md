# PyTorch to ONNX FastAPI Serving with Quantization and Batching

This project demonstrates a comprehensive workflow for deploying a PyTorch image classification model using FastAPI, ONNX Runtime, and various optimizations including ONNX export, static quantization, and asynchronous batch processing. A simple frontend is also provided for easy interaction.

## Features

- **PyTorch Model Integration**: Uses a pre-trained ResNet18 model from `torchvision` for image classification.
- **ONNX Export**: Converts the PyTorch model to ONNX format for optimized inference with ONNX Runtime.
- **Static Quantization**: Applies post-training static quantization (INT8) to the ONNX model for reduced model size and faster inference, leveraging calibration data.
- **FastAPI Server**: Provides a RESTful API for model inference, including single-image predictions and asynchronous batch processing.
- **Asynchronous Batching**: Implements an `asyncio.Queue`-based batch processing mechanism to efficiently handle multiple inference requests.
- **ONNX Runtime Inference**: Utilizes ONNX Runtime for high-performance model execution for both FP32 and INT8 models.
- **Simple Web Frontend**: A single HTML page (`static/index.html`) for uploading images, selecting model variants (FP32/INT8), and displaying top-5 predictions.

## Project Structure

```
ai_engineering/
├── requirements.txt              # Project dependencies
├── models/                       # Contains model-related code
│   ├── __init__.py
│   ├── pytorch_model.py          # PyTorch ResNet18 model setup and basic inference
│   ├── onnx_export.py            # Script for exporting PyTorch model to ONNX
│   └── quantization.py           # Script for static quantization of ONNX model
├── serving/                      # Contains FastAPI application and serving logic
│   ├── __init__.py
│   ├── onnx_inference.py         # ONNX Runtime inference wrapper
│   ├── batch_processor.py        # Asynchronous batch processing implementation
│   └── fastapi_app.py            # FastAPI application with API endpoints
└── static/                       # Static files for the web frontend
    └── index.html                # Single-page HTML frontend for image classification
```

## Setup and Running Guide

Follow these steps to set up and run the project locally.

### 1. Clone the repository (if not already done)

```bash
# Assuming you are in the parent directory where ai_engineering will reside
git clone <repository_url>
cd ai_engineering
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Export ONNX Model

First, export the pre-trained PyTorch ResNet18 model to ONNX format. This will create `models/resnet18.onnx`.

```bash
python -c "from models.onnx_export import export_onnx; export_onnx()"
```

### 4. Quantize ONNX Model (Optional, but Recommended)

Perform static quantization on the ONNX model. This will create `models/resnet18_quantized.onnx`.
_Note: This step requires a calibration dataset. For simplicity, the `quantization.py` script uses synthetic data, but for real-world applications, you would use a representative dataset._

```bash
python -c "from models.quantization import quantize_static_onnx; quantize_static_onnx()"
```

### 5. Start the FastAPI Server

Run the FastAPI application using Uvicorn.

```bash
python serving/fastapi_app.py
```

The server will start, typically on `http://127.0.0.1:8000`. You will see messages indicating that the models are loaded and the batch processor has started.

### 6. Access the Web Frontend

Open your web browser and navigate to `http://127.0.0.1:8000`.

- You can now drag and drop an image or click to upload one.
- Select between "FP32 (ONNX)" and "INT8 (Quantized)" models for inference.
- Click "Classify" to get top-5 predictions and inference latency.

### 7. Interact with the API (Optional)

You can also interact directly with the FastAPI endpoints using tools like `curl` or Postman, or by visiting the interactive API documentation at `http://127.0.0.1:8000/docs`.

#### Example: Single Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg;type=image/jpeg"
```

#### Example: List Models

```bash
curl "http://127.0.0.1:8000/models"
```
