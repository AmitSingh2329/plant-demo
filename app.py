from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import io

app = Flask(__name__)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Define ResNet9 Model
# -----------------------------
import torch.nn as nn

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]

    if pool:
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)

        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)

        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)

        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)

        out = self.res2(out) + out

        out = self.classifier(out)

        return out


# -----------------------------
# Class Names
# -----------------------------
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___healthy"
]

# -----------------------------
# Load Model
# -----------------------------
model = ResNet9(3, len(classes))

model.load_state_dict(
    torch.load("plant_disease_model.pth", map_location=device)
)

model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "Plant Disease Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["file"]

    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    result = {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

    return jsonify(result)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)