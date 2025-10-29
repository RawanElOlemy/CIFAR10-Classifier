import torch
import torch.nn as nn
import torch.nn.functional as F

# CIFAR-10 class names
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Simple CNN Model (same as training architecture)
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load saved model
def load_model(model_path="models/cifar10_model.pth"):
    model = CIFAR10Model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Model file not found. Please train and save it first.")
    return model

# Predict class
def predict_class(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASSES[predicted.item()]
