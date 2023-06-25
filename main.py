from flask import Flask, request

import pickle

import numpy as np


app = Flask(__name__)

#http://localhost:5000/api_predict

model_pkl = pickle.load(open("final_model.sav" , "rb"))

@app.route('/api_predict', methods = ["GET" , "POST"])

def api_predict():
    if request.method == "GET":
        return "Please Send POST Request"
    elif request.method == "POST":
        data = request.get_json()
        
        Material = data["Material"]
        Additive = data["Additive"]
        Ash = data["Ash"]
        Plasticizer = data["Plasticizer"]
        Formulation = data["Formulation"]
        
        in1 = np.array([[Material, Additive, Ash,Plasticizer, Formulation]])
        
        predictions = model_pkl.predict(in1)
        
        return str(predictions)
    
app.run()




