from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Tillåter anrop från alla domäner

# Ladda modellen
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return jsonify({"message": "API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        sentiment = model.predict([text])[0]
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        print(f"Error in prediction: {e}")  # Logga felet
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)