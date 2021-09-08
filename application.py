# libraries
from scripts.predict import simons_response
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle 
import json

#items to load
model = load_model("artifacts/model.h5")


application = Flask(__name__)

@application.route("/")
def home():
    return render_template("index.html")


@application.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    model = load_model("artifacts/model.h5")
    res = simons_response(msg,model)
    return res


if __name__ == "__main__":
    application.run()

