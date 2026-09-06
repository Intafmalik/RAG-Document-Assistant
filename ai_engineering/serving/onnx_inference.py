import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import onnxruntime as ort


class ONNXModelWrapper:
    """Wrapper for ONNX Runtime model inference with FP32 and INT8 support."""

    def __init__(self, model_path: str, use_quantized: bool = False):
        self.use_quantized = use_quantized
        if use_quantized:
            # Try quantized path first, fall back to original
            fallback = model_path.replace("_quantized.onnx", ".onnx")
            self.model_path = model_path if os.path.exists(model_path) else fallback
        else:
            self.model_path = model_path

        self.session = ort.InferenceSession(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image: "PIL.Image.Image") -> np.ndarray:
        """Preprocess a PIL Image to normalized float32 numpy array."""
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

        preprocess = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = preprocess(image)
        return tensor.numpy().astype(np.float32)

    async def predict(self, image: "PIL.Image.Image") -> Dict[str, Any]:
        """Run inference on a single PIL Image and return predictions."""
        start = time.time()

        input_data = self.preprocess(image).reshape(1, 3, 224, 224).astype(np.float32)

        outputs = self.session.run(None, {"input": input_data})
        logits = outputs[0]

        # Softmax to probabilities
        probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))

        # Top-5
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_probs = probs[top5_idx]

        latency_ms = (time.time() - start) * 1000

        return {
            "logits": logits,
            "probabilities": probs.tolist(),
            "top5_indices": top5_idx.tolist(),
            "top5_probabilities": top5_probs.tolist(),
            "latency_ms": latency_ms,
            "model_path": self.model_path,
        }

    def predict_sync(self, image: "PIL.Image.Image") -> Dict[str, Any]:
        """Synchronous version of predict()."""
        start = time.time()

        input_data = self.preprocess(image).reshape(1, 3, 224, 224).astype(np.float32)

        outputs = self.session.run(None, {"input": input_data})
        logits = outputs[0]

        probs = np.exp(logits[0]) / np.sum(np.exp(logits[0]))

        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_probs = probs[top5_idx]

        latency_ms = (time.time() - start) * 1000

        return {
            "logits": logits,
            "probabilities": probs.tolist(),
            "top5_indices": top5_idx.tolist(),
            "top5_probabilities": top5_probs.tolist(),
            "latency_ms": latency_ms,
            "model_path": self.model_path,
        }