from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("stock_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        open_price = float(request.form["open_price"])
        high_price = float(request.form["high_price"])
        low_price = float(request.form["low_price"])
        volume = float(request.form["volume"])
        previous_close = float(request.form["previous_close"])

        # Prepare input
        input_data = np.array([[
            open_price,
            high_price,
            low_price,
            volume,
            previous_close
        ]])

        # Prediction
        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Please enter valid numbers."
        )


if __name__ == "__main__":
    app.run(debug=True)