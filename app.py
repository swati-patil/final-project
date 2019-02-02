from flask import Flask, render_template, redirect, jsonify
import requests
import json
import pprint
from bson.json_util import dumps
import datetime, time
import pandas as pd
import keras
from keras.preprocessing import image
from keras import backend as K
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle

# Flask Setup
app = Flask(__name__)

#home route
@app.route("/")
def index():
	return render_template("index.html")

@app.route("/predict/<price>/<category>")
def display_prediction(price=None, category=None):
	if category is None:
		category = 10 #default for Litrature and fiction
	
	n_price = float(price)
	n_category = int(category)
	data = np.array([price, category])
	data = data.reshape(-1, 2)

	loaded_model = pickle.load(open("random-forest-model.sav", 'rb'))
	result = loaded_model.predict(data)
	print(result)
	ret_data = {'predicted_val' : result}
	return dumps(ret_data)

if __name__ == "__main__":
    app.run(debug=True)