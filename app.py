import os
from flask import Flask, request, jsonify
from predict import predict_skin_disease

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return jsonify({"message": "Skin Disease Detection API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use png/jpg/jpeg"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        result = predict_skin_disease(file_path)

        return jsonify({
            "success": True,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "description": result["description"],
            "precautions": result["precautions"],
            "warning": result["warning"],
            "note": "This AI result is only for assistance and not a final medical diagnosis."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

