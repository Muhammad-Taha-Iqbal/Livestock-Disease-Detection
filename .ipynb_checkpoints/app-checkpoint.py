from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
from datetime import datetime
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input  

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

CORS(app)

# LOAD TREATMENTS
with open("treatments.json", "r") as f:
    TREATMENTS = json.load(f)

# LOAD MODELS
MODELS = {
    "cow": tf.keras.models.load_model("AImodel_VGG16_Cow.h5"),
    "goat": tf.keras.models.load_model("AImodel_VGG16_Goat.h5"),
    "camel": tf.keras.models.load_model("AImodel_VGG16_Camel.h5"),
    "chicken": tf.keras.models.load_model("AImodel_VGG162_Chicken.h5"),
}

# CLASS LABELS
CLASS_NAMES = {
    "cow": ["healthy cows", "foot and mouth", "lumpy"],
    "chicken": ["anemia virus", "bumble foot", "fowl pox disease", "healthy chicken"],
    "camel": ["eye infection", "foot and mouth", "healthy camel", "scabies"],
    "goat": ["goatpox", "healthy goat", "soremouth"]
}

# IMAGE PREPROCESSING
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)   
    img = preprocess_input(img)              
    img = np.expand_dims(img, axis=0)
    return img

# SAVE RESULT
def save_result(animal, disease, confidence):
    data = {
        "animal": animal,
        "disease": disease,
        "confidence": confidence,
        "time": str(datetime.now())
    }

    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except:
        results = []

    results.append(data)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

# =========================
# ⭐ NEW: HISTORY API
# =========================
@app.route("/history", methods=["GET"])
def history():
    try:
        with open("results.json", "r") as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify([])

# PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        animal = request.form.get("animal", "").lower().strip()
        if animal not in MODELS:
            return jsonify({"error": f"Invalid animal type: {animal}"}), 400

        image = request.files["image"]

        processed_img = preprocess_image(image)

        model = MODELS[animal]
        preds = model.predict(processed_img)

        class_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        disease = CLASS_NAMES[animal][class_index]

        disease_key = disease.lower().strip()

        treatment_data = TREATMENTS.get(animal, {}).get(disease_key, {
            "treatment": "No treatment found",
            "precautions": "No precautions found"
        })

        response = {
            "status": "success",
            "animal": animal,
            "disease": disease,
            "confidence": round(confidence, 4),
            "treatment": treatment_data.get("treatment"),
            "precautions": treatment_data.get("precautions")
        }

        save_result(animal, disease, round(confidence, 4))

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# FRONTEND ROUTES
@app.route("/")
def home():
    return send_from_directory("frontend", "welcome.html")

@app.route("/<path:path>")
def serve_page(path):
    return send_from_directory("frontend", path)


if __name__ == "__main__":
    app.run(debug=True, port=5001)