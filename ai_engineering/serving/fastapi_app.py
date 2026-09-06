import asyncio
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from models.pytorch_model import get_model, predict as pt_predict
from models.onnx_export import load_onnx_model, run_onnx_inference
from models.quantization import quantize_static_onnx
from serving.onnx_inference import ONNXModelWrapper
from serving.batch_processor import BatchInferenceProcessor

import numpy as np


app = FastAPI(title="ResNet18 Image Classification Tutorial")

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model instances
fp32_model: Optional[ONNXModelWrapper] = None
int8_model: Optional[ONNXModelWrapper] = None
batch_processor: Optional[BatchInferenceProcessor] = None

# In-memory store for bulk upload tasks: bulk_id -> {tasks: {task_id: status_info}}
bulk_tasks: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def load_models():
    global fp32_model, int8_model, batch_processor

    # Load FP32 ONNX model
    fp32_model = ONNXModelWrapper("models/resnet18.onnx", use_quantized=False)

    # Load quantized (INT8) model if available
    quant_path = "models/resnet18_quantized.onnx"
    if os.path.exists(quant_path):
        int8_model = ONNXModelWrapper(quant_path, use_quantized=True)
        print(f"✅ Loaded INT8 quantized model from {quant_path}")
    else:
        print("⚠️  INT8 quantized model not found. Run export + quantization first.")

    # Initialize batch processor with shared INT8 session (or FP32)
    if int8_model:
        batch_processor = BatchInferenceProcessor(
            session=int8_model.session,
            batch_size=4,
            timeout_sec=0.5,
        )
    else:
        batch_processor = BatchInferenceProcessor(
            session=fp32_model.session,
            batch_size=4,
            timeout_sec=0.5,
        )
    await batch_processor.start()
    print("✅ Batch inference processor started.")


@app.on_event("shutdown")
async def cleanup():
    global batch_processor
    if batch_processor:
        await batch_processor.stop()


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the frontend HTML page."""
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Frontend not found</h1>")


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    """Run single-image inference using the selected model."""
    global fp32_model, int8_model

    if not fp32_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Read uploaded image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Choose model: INT8 if available and selected, otherwise FP32
    model = int8_model if int8_model else fp32_model
    result = model.predict_sync(image)

    # Map top-5 indices to ImageNet class names
    from torchvision.models import ResNet18_Weights
    class_names = ResNet18_Weights.DEFAULT.meta["categories"]
    mapped = []
    for idx, prob in zip(result["top5_indices"], result["top5_probabilities"]):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        mapped.append({"class": name, "confidence": round(prob * 100, 2)})

    return {
        "success": True,
        "model": "INT8 (Quantized)" if int8_model else "FP32 (ONNX)",
        "predictions": mapped,
        "latency_ms": round(result["latency_ms"], 2),
    }


@app.post("/predict/batch")
async def predict_batch_submit(file: UploadFile = File(...)):
    """Submit an image for batch inference via the async queue.

    Returns a task_id that can be polled with GET /predict/batch/{task_id}.
    """
    global batch_processor

    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not ready")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess using FP32 model's method
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image)
    input_data = tensor.numpy().astype(np.float32).reshape(1, 3, 224, 224)

    # Submit to queue
    future = asyncio.get_event_loop().create_future()
    task_id = str(uuid.uuid4())

    await batch_processor.queue.put((input_data, future))

    return {
        "success": True,
        "task_id": task_id,
        "status": "queued",
        "message": "Image submitted to batch inference queue",
    }


@app.get("/predict/batch/{task_id}")
async def predict_batch_poll(task_id: str):
    """Poll for batch inference result.

    Returns the result if ready, or {status: "pending"} if still processing.
    """
    global batch_processor

    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not ready")

    # Check if we have a result for this task_id
    if task_id in batch_processor._results:
        future = batch_processor._results[task_id]
        if future.done():
            try:
                result, latency = future.result()
                # Map to ImageNet class names
                from torchvision.models import ResNet18_Weights
                class_names = ResNet18_Weights.DEFAULT.meta["categories"]
                probs = np.exp(result[0]) / np.sum(np.exp(result[0]), axis=1, keepdims=True)
                top5_idx = np.argsort(-probs[:, 0])[:, :5].flatten().tolist()
                top5_probs = np.sort(-probs[:, 0])[:, :5][::-1].flatten().tolist()

                mapped = []
                for idx, prob in zip(top5_idx, top5_probs):
                    name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
                    mapped.append({"class": name, "confidence": round(float(prob) * 100, 2)})

                return {
                    "status": "complete",
                    "task_id": task_id,
                    "predictions": mapped,
                    "latency_ms": round(float(latency), 2),
                }
            except Exception as e:
                return {
                    "status": "error",
                    "task_id": task_id,
                    "error": str(e),
                }

    # Task not found or still processing
    return {"status": "pending", "task_id": task_id}


@app.get("/models")
async def list_models():
    """List available model configurations."""
    return {
        "fp32": {
            "path": "models/resnet18.onnx",
            "description": "Full precision ONNX model",
        },
        "int8": {
            "path": "models/resnet18_quantized.onnx"
            if os.path.exists("models/resnet18_quantized.onnx")
            else None,
            "description": "Static quantized (INT8) model",
            "available": os.path.exists("models/resnet18_quantized.onnx"),
        },
    }


@app.post("/predict/bulk")
async def predict_bulk(files: List[UploadFile] = File(...)):
    """Upload multiple images for async batch inference.

    Returns immediately with a bulk_id and per-file task_ids.
    Processing happens in the background via asyncio.Queue.
    Poll GET /predict/bulk/{bulk_id} for status updates.
    """
    global batch_processor

    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not ready")

    bulk_id = str(uuid.uuid4())
    tasks = {}

    # ImageNet class names
    from torchvision.models import ResNet18_Weights
    class_names = ResNet18_Weights.DEFAULT.meta["categories"]

    for file in files:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            tasks[file.filename] = {
                "status": "error",
                "error": "Invalid image file",
            }
            continue

        # Preprocess
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
        preprocess = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = preprocess(image)
        input_data = tensor.numpy().astype(np.float32).reshape(1, 3, 224, 224)

        task_id = str(uuid.uuid4())

        # Track task
        tasks[file.filename] = {
            "task_id": task_id,
            "status": "queued",
            "result": None,
        }

        # Run inference in background (non-blocking)
        asyncio.create_task(
            _process_and_store(bulk_id, file.filename, input_data, class_names)
        )

    bulk_tasks[bulk_id] = {
        "tasks": tasks,
        "created_at": time.time(),
    }

    return {
        "success": True,
        "bulk_id": bulk_id,
        "tasks": tasks,
        "total_files": len(files),
        "queued": sum(1 for t in tasks.values() if t["status"] == "queued"),
        "errors": sum(1 for t in tasks.values() if t["status"] == "error"),
        "status_url": f"/predict/bulk/{bulk_id}",
    }


async def _process_and_store(
    bulk_id: str,
    filename: str,
    input_data: np.ndarray,
    class_names: list,
):
    """Process a single image via ONNX Runtime and store results."""
    global fp32_model, int8_model

    try:
        model = int8_model if int8_model else fp32_model
        if not model:
            raise RuntimeError("No model loaded")

        start = time.time()
        outputs = model.session.run(None, {"input": input_data})
        logits = outputs[0]
        latency = (time.time() - start) * 1000

        probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_probs = probs[top5_idx]

        mapped = []
        for idx, prob in zip(top5_idx, top5_probs):
            name = class_names[int(idx)] if int(idx) < len(class_names) else f"class_{int(idx)}"
            mapped.append({"class": name, "confidence": round(float(prob) * 100, 2)})

        bulk_tasks[bulk_id]["tasks"][filename] = {
            "task_id": bulk_tasks[bulk_id]["tasks"][filename]["task_id"],
            "status": "complete",
            "predictions": mapped,
            "latency_ms": round(latency, 2),
        }
    except Exception as e:
        bulk_tasks[bulk_id]["tasks"][filename]["status"] = "error"
        bulk_tasks[bulk_id]["tasks"][filename]["error"] = str(e)


@app.get("/predict/bulk/{bulk_id}")
async def predict_bulk_status(bulk_id: str):
    """Check the processing status of a bulk upload.

    Returns overall progress and per-file results as they complete.
    """
    if bulk_id not in bulk_tasks:
        raise HTTPException(status_code=404, detail="Bulk task not found")

    bulk = bulk_tasks[bulk_id]
    tasks = bulk["tasks"]

    total = len(tasks)
    completed = sum(1 for t in tasks.values() if t["status"] == "complete")
    errors = sum(1 for t in tasks.values() if t["status"] == "error")
    pending = total - completed - errors

    return {
        "bulk_id": bulk_id,
        "progress": {
            "total": total,
            "completed": completed,
            "errors": errors,
            "pending": pending,
            "percent": round((completed + errors) / total * 100, 1) if total > 0 else 0,
        },
        "tasks": tasks,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)