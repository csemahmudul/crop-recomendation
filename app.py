from flask import Flask, request, render_template
import numpy as np
import pickle

# Create flask app
flask_app = Flask(__name__)

# Load the model
model = pickle.load(open("joy.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html", prediction_text="")  # Blank prediction initially

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction[0]))

if __name__ == "__main__":
    flask_app.run(debug=True)
