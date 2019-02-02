from flask import Flask, render_template, redirect, jsonify
# import mongo lib
#import pymongo
import requests
import json
#from process_response import trim_response
import pprint
#from bson.json_util import dumps
import datetime, time
#from datetime import timedelta
import pandas as pd
import keras
from keras.preprocessing import image
from keras import backend as K
import numpy as np

# Flask Setup
app = Flask(__name__)
model=None

#home route
@app.route("/")
def index():
	return render_template("index.html")

#@app.route("/all")
#def all():
	#response_data = pd.read_csv('sorted_by_category.csv')
    #response_json = response_data.to_json()

	#return response_data.to_json()

def load_model():
    global model
    model = keras.models.load_model("deep_learning.py")


def deep_learn_predict():
	data = np.array([14.95, 10])
	print(data.shape)
	data = data.reshape(-1, 2)
	pred = model.predict_classes(data)
	pred_labels = label_encoder.inverse_transform(pred)
	print(pred_labels)



@app.route("/filter")

def user_filter(cat=None):
	user_fi = {}

	if (cat is not None):
        #if no category selected
		category = {'$gt' : float(cat)}
		user_fi['category'] = category
	
	print(user_fi)
	if (user_fi is not None):
        #if there is a category selected, go through the csv and grab the data for just that category 
		res = filter_response.find(user_fi)
	else:
        #else call the whole csv data
		res = filter_response.find()



if __name__ == "__main__":
    app.run(debug=True)