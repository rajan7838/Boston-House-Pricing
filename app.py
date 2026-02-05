import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
reg_model = pickle.load(open("lr.pkl", "rb"))
reg_scaler = pickle.load(open("scaler.pkl", "rb"))


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# ----------------------------------------
# API Prediction Route (For Postman / JSON)
# ----------------------------------------
@app.route("/predict_api", methods=["POST"])
def predict_api():

    # FIX: request.json (not request.jason)
    data = request.json["data"]

    # Convert values to numpy array
    input_array = np.array(list(data.values())).reshape(1, -1)

    # Scale input
    scaled_data = reg_scaler.transform(input_array)

    # Predict
    output = reg_model.predict(scaled_data)

    return jsonify({"prediction": output[0]})


# ----------------------------------------
# HTML Form Prediction Route
# ----------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Take input from form values
    form_data = [float(x) for x in request.form.values()]

    # Convert to array
    input_array = np.array(form_data).reshape(1, -1)

    # Scale input
    scaled_data = reg_scaler.transform(input_array)

    # Predict output
    prediction = reg_model.predict(scaled_data)[0]

    return render_template(
        "home.html",
        prediction_text=f"🏠 Predicted House Price: {prediction:.2f} (in $1000s)"
    )


# Run App
if __name__ == "__main__":
    app.run(debug=True)
