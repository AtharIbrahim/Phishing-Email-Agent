# from flask import Flask, request, jsonify
# import joblib

# # Load trained model
# model = joblib.load("phishing_model.pkl")

# # Initialize Flask app
# app = Flask(__name__)

# @app.route("/detect", methods=["POST"])
# def detect_phishing():
#     data = request.get_json()
#     email_text = data.get("text", "")

#     # Make prediction
#     prediction = model.predict([email_text])[0]
#     result = "Phishing" if prediction == 1 else "Legitimate"

#     return jsonify({"result": result})

# if __name__ == "__main__":
#     app.run(debug=True)




# After running server
# app.py
# run this command on cmd
# curl -X POST http://127.0.0.1:5000/detect -H "Content-Type: application/json" --data "{\"text\": \"Your bank account is locked! Click here to verify\"}"


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer
MODEL_PATH = "phishing_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None
    print("❌ Error: Model or vectorizer file not found!")

@app.route("/detect", methods=["POST"])
def detect():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded properly"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"result": "Error: No email text provided"}), 400

    email_text = data["text"]

    # Convert email text to numerical format
    email_vectorized = vectorizer.transform([email_text])

    # Predict phishing or legitimate
    prediction = model.predict(email_vectorized)[0]

    # Convert prediction to human-readable format
    result = "Phishing" if prediction == 1 else "Legitimate"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
