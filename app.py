# libraries
from scripts.predict import simons_response
from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok
from keras.models import load_model
import pickle 
import json

#items to load
model = load_model("artifacts/model.h5")


app = Flask(__name__)
#run_with_ngrok(app) -Use this option if you have ngrok and you want to expose your chatbot to the real world

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    model = load_model("artifacts/model.h5")
    res = simons_response(msg,model)
    return res


if __name__ == "__main__":
    app.run()

#print(simons_response("Hello my name is levi"))
