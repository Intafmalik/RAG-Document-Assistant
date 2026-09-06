import os
import torch
import numpy as np
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


class CalibrationDataReader:
    """Feeds calibration data to onnxruntime quantization."""

    def __init__(self, calibration_data: list, input_name: str = "input"):
        self.calibration_data = calibration_data
        self.input_name = input_name
        self.current_index = 0

    def get_next(self):
        if self.current_index >= len(self.calibration_data):
            return None
        data = self.calibration_data[self.current_index]
        self.current_index += 1
        return {self.input_name: data}


def quantize_static_onnx(
    model_path: str = "models/resnet18.onnx",
    quantized_path: str = "models/resnet18_quantized.onnx",
    calibration_images: list = None,
):
    """Quantize an ONNX model using static quantization with calibration."""
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod

    if calibration_images is None:
        calibration_images = [torch.randn(3, 224, 224) for _ in range(10)]

    calib_data = []
    for img in calibration_images:
        if isinstance(img, torch.Tensor):
            img_np = img.numpy().astype(np.float32)
            img_np = np.expand_dims(img_np, axis=0)  # add batch dim
        else:
            img_np = img.astype(np.float32) if isinstance(img, np.ndarray) else np.random.randn(1, 3, 224, 224).astype(np.float32)
        calib_data.append(img_np)

    reader = CalibrationDataReader(calib_data, input_name="input")

    print(f"Quantizing ONNX model with {len(calib_data)} calibration samples...")
    quantize_static(
        model_input=model_path,
        model_output=quantized_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )

    print(f"Quantized model saved to {quantized_path}")


class QuantFormat:
    QDQ = "QDQ"