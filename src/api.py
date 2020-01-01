import flask
from tensorflow import keras
import numpy as np
import pandas as pd
from flask import request, jsonify


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def test():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/api/prediction', methods=['GET'])
def precict():
    if 'sample' in request.args:
        sample = request.args['sample']
    else:
        return "Error: No sample field found. "
    #x_test, y_test = test_sample.iloc[:, :-1], test_sample.iloc[:, -1]
    #y_pred = model.predict(x_test)
    #model = keras.models.load_model('CNN-.h5') # todo: get right model
    return "test"

app.run()
