from flask import Flask, request, jsonify
from flask_cors import CORS
import with_antiSpoofing  # your anti-spoofing script

app = Flask(__name__)
CORS(app)  # Allow requests from Angular or any frontend

@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods =['POST'])
def predict():
    # check if file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily
    filepath = "temp_input.jpg"
    file.save(filepath)

    # Call your antispoofing + face recognition pipeline
    try:
        result = with_antiSpoofing.run_model(filepath)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # disable reloader to avoid WinError 10038 on Windows
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
