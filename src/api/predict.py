import torch
from torchvision import transforms, models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10 label names
LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Preprocessing identical to validation transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

def load_model(path="model.pth"):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def predict(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = outputs.max(1)
        label = LABELS[predicted.item()]
    return label