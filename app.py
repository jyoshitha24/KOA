from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import cv2
from torchvision import transforms
from model import load_model

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained model
model = load_model("model2.pth")

# Severity labels
severity_levels = ["Grade 0-normal", "Grade 1-doubtful", "Grade 2-mild", "Grade 3-moderate", "Grade 4-severe"]

# Preprocessing transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read image and apply transform
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        result = severity_levels[predicted_class]
        return jsonify({"severity": result})
    
    except Exception as e:
        return jsonify({"error": f"Error processing the image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

