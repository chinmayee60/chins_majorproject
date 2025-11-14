from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import io

app = Flask(__name__)

# --- Configuration & Model Loading ---
MODEL_PATH = 'backend\models\wheat_stem_model_final.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # --- Redefine CNN_NeuralNet architecture (Crucial for torch.load to work) ---
    import torch.nn as nn
    class CNN_NeuralNet(nn.Module):
        def __init__(self, in_channels, num_classes):
            super(CNN_NeuralNet, self).__init__()
            self.network = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        def forward(self, xb):
            return self.network(xb)

    # Load the entire model
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    print(f"✅ Wheat Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Error loading Wheat Model: {e}")
    model = None

# --- Get Classes (Sorted list of 15 classes from your notebook output) ---
WHEAT_CLASSES = [
    'Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot', 
    'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite', 
    'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust'
]

# --- Preprocessing Transform (Matching your val/test transform) ---
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # Normalization was NOT applied in your test code for this model, so we omit it here.
])


@app.route('/predict_wheat', methods=['POST'])
def predict_wheat():
    if model is None:
        return jsonify({"error": "Wheat Model not loaded"}), 500
        
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
        
        disease_name = WHEAT_CLASSES[predicted_idx.item()]
        confidence = predicted_prob.item() * 100
        
        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)