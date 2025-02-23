# app.py
from flask import Flask, request, send_from_directory, render_template, jsonify
import os
import cv2
import numpy as np
from image_processor import extract_price_data, detect_pattern, markup_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (or specify origins like CORS(app, resources={r"/upload": {"origins": "http://localhost:3000"}}))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        # Extract prices and detect pattern
        try:
            prices = extract_price_data(filename)
            troughs, peaks = detect_pattern(prices)

            # Determine pattern and decision (simplified logic)
            if len(troughs) >= 2 and troughs[1] - troughs[0] < 50:  # Example: Double Bottom within 50 points
                pattern = "Double Bottom"
                decision = "Buy"
            elif len(peaks) >= 2 and peaks[1] - peaks[0] < 50:  # Example: Double Top within 50 points
                pattern = "Double Top"
                decision = "Sell"
            else:
                pattern = "Flat Trend"
                decision = "Hold"

            # Markup the image
            output_image = markup_image(filename, prices, troughs, peaks)
            output_path = os.path.join(STATIC_FOLDER, 'output.png')
            cv2.imwrite(output_path, output_image)

            # Return JSON response with pattern, decision, and image URL
            response = {
                "pattern": pattern,
                "decision": decision,
                "image_url": f"http://localhost:5000/static/output.png"
            }
            return jsonify(response), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)