import torch
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


CLASS_NAMES = [
    " airplane", " automobile", " bird", " cat", " deer",
    " dog", " frog", " horse", " ship", " truck",
]


def get_model():
    """Load pre-trained ResNet18 in eval mode."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    """Preprocess a PIL Image for ResNet18 input."""
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)


def predict(image: Image.Image, model=None):
    """Run inference on a single PIL Image and return top-5 predictions."""
    if model is None:
        model = get_model()

    tensor = preprocess_image(image).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        logits = model(tensor)

    probs = torch.softmax(logits, dim=1)[0]
    top5_probs, top5_idx = torch.topk(probs, 5)

    results = []
    for prob, idx in zip(top5_probs, top5_idx):
        class_idx = idx.item()
        confidence = prob.item() * 100
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"class_{class_idx}"
        results.append({
            "class": class_name,
            "confidence": round(confidence, 2),
        })

    return results