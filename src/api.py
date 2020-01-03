import flask
from tensorflow import keras
import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from flask import request, jsonify
from pandas.io.json import json_normalize


# function taken from: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def test():
    return "<h1<Test</h1><p>API is live.</p>"

@app.route('/api/prediction', methods=['POST'])
def precict():


    # Validate the request body contains JSON
    if request.is_json:

        # Parse the JSON into a Python dictionary
        req = request.get_json()
        sample_df = json_normalize(req)

        timesteps = 40
        #sample_df = sample_df.drop(["TIMESTAMP"], axis=1)
        sample_df = sample_df.astype(float)

        x_test, y_test = sample_df.iloc[:, :-1], sample_df.iloc[:, -1]

        n_features = 83
        x_test_reshaped = x_test.values.reshape(x_test.shape[0], timesteps + 1, n_features)


        optimizer = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004)

        model = keras.models.load_model('models/CNN-1.h5', compile=False) # todo: get right model
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])


        y_pred = model.predict(x_test_reshaped)
        y_class = y_pred.argmax(axis=-1)
        y_class = y_class + 1

        y_pred_pd = pd.DataFrame(y_class, columns=["class"])
        y_test_pd = pd.DataFrame(y_test.tolist(), columns=["class"])

        # activity_map = {0: "no activity", 1: "Act01", 2: "Act02", 3: "Act03", 4: "Act04", 5: "Act05", 6: "Act06", 7: "Act07", 8: "Act08",
        #                 9: "Act09",  10: "Act10", 11: "Act11", 12: "Act12", 13: "Act13", 14: "Act14",  15: "Act15",
        #                 16: "Act16", 17: "Act17", 18: "Act18", 19: "Act19", 20: "Act20", 21: "Act21", 22: "Act22",
        #                 23: "Act23", 24: "Act24"}
        activity_map = {0: "no activity", 1: "Take medication", 2: "Prepare breakfast", 3: "Prepare lunch", 4: "Prepare dinner",
                        5: "Breakfast", 6: "Lunch", 7: "Dinner", 8: "Eat a snack", 9: "Watch TV",  10: "Enter the SmartLab",
                        11: "Play a videogame", 12: "Relax on the sofa", 13: "Leave the SmartLab", 14: "Visit in the SmartLab",
                        15: "Put waste in the bin", 16: "Wash hands", 17: "Brush teeth", 18: "Use the toilet", 19: "Wash dishes",
                        20: "Put washin into the washing machine", 21: "Work at the table", 22: "Dressing", 23: "Go to the bed",
                        24: "Wake up"}
        predicted_class = y_pred_pd["class"].map(activity_map)
        y_test_pd = y_test_pd.astype(float)
        actual_class = y_test_pd["class"].map(activity_map)

        prediction_result = "The new data point is predicted to be the activity {} ({}). The ground truth activity is {} ({}). ".format(predicted_class[0], y_class[0], actual_class[0], int(y_test[0]))
        if(y_class[0] == int(y_test[0])):
            prediction_result += "The system predicted correctly! "
        else:
            prediction_result += "The system predicted wrong! "

        print(prediction_result)

        # Return a string along with an HTTP status code
        return prediction_result, 200

    else:

        # The request body wasn't JSON so return a 400 HTTP status code
        return "Request was not JSON", 400


app.run()
