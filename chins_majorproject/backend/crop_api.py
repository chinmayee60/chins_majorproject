from flask import Flask, request, jsonify
import pickle
import numpy as np
import random # Used for dummy prediction if model fails

app = Flask(__name__)

# --- Configuration & Model Loading ---
MODEL_PATH = 'backend/models/crop_recommendation_model.pkl'

# Example Crop Classes (Must match the labels used during Scikit-learn training)
CROP_CLASSES = ['Rice', 'Maize', 'Jute', 'Coffee', 'Cotton', 'Sugarcane', 'Millet', 'Wheat', 'Barley', 'Pulses']

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Crop Recommendation Model loaded successfully.")
    is_dummy = False
except Exception as e:
    print(f"❌ Error loading Crop Model (using dummy): {e}")
    # Dummy model logic for demonstration if .pkl is missing
    class DummyModel:
        def predict(self, data):
            # This mimics a classification model returning a class index
            return [random.randint(0, len(CROP_CLASSES) - 1)]
    model = DummyModel()
    is_dummy = True


@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    
    if not data or not all(key in data for key in ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']):
        return jsonify({"error": "Missing required input data (N, P, K, temperature, humidity, pH, rainfall)"}), 400

    try:
        # Prepare data for prediction (Scikit-learn requires 2D array)
        input_data = np.array([
            data['N'], data['P'], data['K'], data['temperature'], 
            data['humidity'], data['pH'], data['rainfall']
        ]).reshape(1, -1)
        
        prediction_index = model.predict(input_data)[0]
        
        # prediction_index is the class index from the model output
        recommended_crop = CROP_CLASSES[prediction_index] 
        
        return jsonify({
            "crop_recommendation": recommended_crop
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(port=5002, debug=True)