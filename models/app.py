from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

# Load the deep learning model
model_path = "./deep_learning_model.h5" 
deep_learning_model = tf.keras.models.load_model(model_path)

# Load the scaler
scaler_path = "./scaler.pkl"  
scaler = joblib.load(scaler_path)

@app.route("/")
def index():
    """Serve the HTML form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle predictions based on user input."""
    try:
        # Get data from the request
        input_data = request.json
        feature_values = np.array(input_data["features"]).reshape(1, -1)  # Ensure 2D array

        # Scale the input features
        scaled_features = scaler.transform(feature_values)

        # Make predictions
        probabilities = deep_learning_model.predict(scaled_features).tolist()[0]
        prediction = int(np.argmax(probabilities))  # Get the class with the highest probability

        # Return the prediction and probabilities
        return jsonify({
            "prediction": prediction,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
