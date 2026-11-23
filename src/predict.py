"""PyTorch MobileNetV2 inference using metadata embedded in the checkpoint."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'models' / 'agrovision_mobilenetv2.pth'

@lru_cache(maxsize=1)
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'Model not found: {CHECKPOINT}')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if checkpoint.get('model_name') != 'mobilenet_v2':
        raise ValueError('Expected a MobileNetV2 checkpoint')
    names = checkpoint['class_names']
    num_classes = int(checkpoint['num_classes'])
    if len(names) != num_classes:
        raise ValueError('Class mapping does not match model outputs')
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    size = int(checkpoint['image_size'])
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint['mean'], std=checkpoint['std']),
    ])
    return model, preprocess, names, checkpoint

def validate_uploaded_image(image: Image.Image):
    """
    Narrow input safeguard: reject obvious human-face photographs.
    This is NOT a complete tomato-leaf detector; other unrelated objects
    may still pass and require a separately trained/validated image gate.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")

    if cv2 is None:
        raise RuntimeError(
            "Image validation is unavailable because OpenCV is not installed. "
            "Install opencv-python-headless from requirements.txt and restart."
        )

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = str(
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError("Could not load the face-validation model.")

    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) > 0:
        raise ValueError(
            "A human face was detected. AgroVision only supports tomato-leaf "
            "images. Please upload a clear photograph of a tomato leaf."
        )


def predict_image(image: Image.Image):
    if not isinstance(image, Image.Image):
        raise TypeError('predict_image expects a PIL image')
    validate_uploaded_image(image)
    model, preprocess, names, _ = load_model()
    tensor = preprocess(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores = probabilities.tolist()
    best = int(torch.argmax(probabilities).item())
    return {
        'class_name': names[best],
        'confidence': float(scores[best]),
        'probabilities': sorted(
            [{'class_name': name, 'confidence': float(score)} for name, score in zip(names, scores)],
            key=lambda item: item['confidence'], reverse=True
        ),
    }
