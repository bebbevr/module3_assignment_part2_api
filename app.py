from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Tillåter frontend att göra anrop till backend

# Ladda in den tränade modellen
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Tar emot JSON-data
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Förutsäg sentiment
    sentiment = model.predict([text])[0]

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
