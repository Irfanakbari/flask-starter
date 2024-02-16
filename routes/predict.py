# app/routes/predict_routes.py
from PIL import Image
from flask import Blueprint, request, jsonify

from ultralytics import YOLO

predict_bp = Blueprint("predict", __name__)
model = YOLO('yolov8n.pt')


@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "file" in request.files:
        # Decode base64 to image
        photo = request.files["file"]
        photo = Image.open(photo.stream)

        # Perform inference
        output = model.predict(photo)

        # Process the output as needed
        detected_objects = {}

        return jsonify({"result_image": detected_objects, })

    return jsonify({"error": "No photo_base64 provided"}), 400
