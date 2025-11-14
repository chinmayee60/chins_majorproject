from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import io

app = Flask(__name__)

# --- Configuration & Model Loading ---
MODEL_PATH = 'backend/models/plant_disease_cnn.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Redefine CNN Architecture (Crucial for torch.load) ---
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
    
    def forward(self, x):
        return self.pool(self.block(x))

class PlantCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # NOTE: Layer 2 is 64->128, Layer 3 is 128->256, Layer 4 is 256->512 in the notebook, 
        # but the class definition in cell 28 had a typo (128->256 and 256->512 were swapped).
        # We assume the architecture that was successfully trained (cell 6) which must be reconstructed here.
        # Based on the typical CNN structure and the execution count, we use the intended progression.
        self.layer1 = ConvBlock(in_channels, 64, pool=True)
        self.layer2 = ConvBlock(64, 128, pool=True) # Corrected based on standard CNN pattern
        self.layer3 = ConvBlock(128, 256, pool=True)
        self.layer4 = ConvBlock(256, 512, pool=True)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

try:
    # Load the entire model including architecture
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    print(f"✅ Plant Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading Plant Model: {e}")
    model = None

# --- Get Classes (Sorted list of 38 classes from your notebook output) ---
PLANT_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
    'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# --- Preprocessing Transform ---
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/predict_plant', methods=['POST'])
def predict_plant():
    if model is None:
        return jsonify({"error": "Plant Model not loaded"}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        img_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_prob, predicted_idx = torch.max(probabilities, 1)
        
        disease_name = PLANT_CLASSES[predicted_idx.item()]
        confidence = predicted_prob.item() * 100
        
        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)