import torch
import torchvision.transforms as transforms
from PIL import Image
from main import CIFAR10Model  

model = CIFAR10Model()
model.load_state_dict(torch.load('./cifar10_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

img = Image.open("images/cat.jpeg")  
img = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Predicted: {classes[predicted.item()]}")
