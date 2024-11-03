import torch
import torchvision.transforms as transforms
from PIL import Image
from model import EncoderCNN

image_path = "../test.jpg"
image = Image.open(image_path).convert('RGB')

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

image_tensor = preprocess(image).unsqueeze(0)

