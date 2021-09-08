# libraries
from scripts.predict import simons_response
from flask import Flask, render_template, request
#from tensorflow.keras.models import load_model
import pickle 
import json

#items to load
#model = load_model("artifacts/model.h5")


appplication = Flask(__name__)

@appplication.route("/")
def home():
    return render_template("index.html")


@appplication.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    #model = load_model("artifacts/model.h5")
    model = "Hello Cam!"
    res = simons_response(msg,model)
    return res


if __name__ == "__main__":
    appplication.run()

