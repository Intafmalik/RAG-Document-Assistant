import os
import torch
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


def export_onnx(
    model_path: str = "models/resnet18.onnx",
    opset_version: int = 16,
    example_outputs: int = 1,
):
    """Export PyTorch ResNet18 to ONNX format."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    # Create example input (1 x 3 x 224 x 224)
    example_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Export using legacy exporter (no onnxscript needed)
    torch.onnx.export(
        model,
        example_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    print(f"✅ ONNX model exported to {model_path}")
    return model_path


def load_onnx_model(model_path: str = "models/resnet18.onnx"):
    """Load an ONNX model for inference."""
    import onnx
    onnx_model = onnx.load(model_path)
    return onnx_model


def run_onnx_inference(model_path: str = "models/resnet18.onnx", input_tensor: torch.Tensor = None):
    """Run inference using ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    if input_tensor is None:
        input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32).numpy()

    outputs = session.run(None, {input_name: input_tensor.numpy()})
    return outputs[0]