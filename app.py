from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

app = Flask(__name__)

class_names = ['Nu.1', 'Nu.10', 'Nu.100', 'Nu.1000', 'Nu.20', 'Nu.5', 'Nu.50', 'Nu.500']

device = torch.device("cpu")
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("currency_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(request.files['image']).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    return jsonify({'prediction': label})
