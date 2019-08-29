import json
import requests
from Static_Analysis import *
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import time


class Sensor_Emulator(Machine_Learn_Static):
    def __init__(self):
        self.scaler = StandardScaler()  # Normalization all X dataset, you can directly use sklearn
        self.env_dict = dict() # record the encode

    # here create an object of your ML algorithm

    # this is to delete the content of temp
    def delete_content(self,filename):
        f = open(filename, 'w')
        f.truncate()
        f.close()

    # this is to send the write command to a POST server in Node-Red
    def send_command_to_server(self, url):
        json_data = {}
        json_data["signal"] = "write"
        r = requests.post(url, data=json_data)
        print("Now send write command to node-red POST server")

    #######  Now blank you algorithm here ##########
    def logic_regress_fit_tf(self, x_train, y_train, x_test):
        theta = tf.Variable(tf.zeros([x_train.shape[1], 1]))
        theta0 = tf.Variable(tf.zeros([1, 1]))
        #x_train = self.scaler.transform(x_train) # normalization
        x_train = x_train.astype(np.float32)
        y = 1 / (1 + tf.exp(-tf.matmul(x_train, theta) + theta0))

        loss = tf.reduce_mean(- y_train.reshape(-1, 1) * tf.log(y) - (1 - y_train.reshape(-1, 1)) * tf.log(1 - y))
        train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)
        for steps in range(10000):
            sess.run(train)

        w, b = sess.run(theta), sess.run(theta0)
        #x_test = self.scaler.transform(x_test) # normalization
        x_test = x_test.astype(np.float32)
        y_pred = 1 / (1 + np.exp(-(x_test.dot(w)) + b))
        y_pred = y_pred.tolist()
        res = []
        for item in y_pred:
            if item[0] >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return np.array(res)

    # Algorithm 2: Naive Bayes


    ########## end of blank your algorithm
    # dump all the lines of file one-by-one to an array
    def file_to_list(self, file):
        f = open(file)
        line = f.readline()
        all_data = []
        while line:
            line = f.readline()
            line = line.strip('\n')
            all_data.append(line)
        all_data.pop()
        return all_data

    # convert a json file to csv file
    def json_file_to_csv(self, src, dst):
        all_data = self.file_to_list(src)
        self.delete_content(dst)
        f = open(dst, 'a')
        keys= [], []
        first = all_data[0]
        j_item = json.loads(first)
        j_item = dict(j_item)
        keys = list(j_item.keys())
        for i in range(len(keys) - 1):
            f.write(keys[i])
            f.write(",")
        f.write(keys[-1])
        f.write("\n")
        for item in all_data:
            j_item = json.loads(item)
            j_item = dict(j_item)
            value_temp = list(j_item.values())
            for i in range(len(value_temp) - 1):
                f.write(str(value_temp[i]))
                f.write(",")
            f.write(str(value_temp[-1]))
            f.write("\n")
        #        np_val = np_val.astype(int)
        f.close()

    # choose x_train, y_train, x_test, y_test, you can overwrite this function
    def arr_split(self, np_arr):
        X = np_arr[:,: -1]
        Y = np_arr[:,-1]
        encode = preprocessing.LabelEncoder()
        Y_enco = encode.fit_transform(Y)
        x_train, x_test, y_train, y_test = train_test_split(X, Y_enco, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    # analysis and provide the results
    def analysis_and_score(self):
        self.json_file_to_csv("temp", "analysis.csv")
        raw_data = pd.read_csv("analysis.csv")
        print(list(raw_data))
        print(raw_data[1:3])

        raw_data.drop(['RecordID'], axis=1, inplace=True)  # delete the repeated feature
        raw_drop = raw_data.dropna()

        # preprocess and split to x_train, y_train, x_test, y_test
        np_data = raw_drop.values
        # if np_data.shape[0] <= 150:
        #     print("The sample data right now is too short")
        #     return 0.0
        np_data_encode = self.encode_array(np_data)
        x_train, x_test, y_train, y_test = self.arr_split(np_data_encode)
        print(x_test)
        print(y_test)
        #y_pred = self.logic_regress_fit(x_train, y_train, x_test)
        y_pred = self.logic_regress_fit_tf(x_train, y_train, x_test)
        print(y_pred)
        score = self.right_rate(y_pred, y_test)
        print("The right rate of logistic regression using Tensorflow is:", score)
        return score

        # send write signal to server
    def send_command_to_server(self, url):
        json_data = {}
        json_data["signal"] = "write"
        # bin_str = js.dumps(json_data)
        r = requests.post(url, data=json_data)

    def plot_scatter(self, x, y):
        plt.scatter(x, y, alpha=1, color='b')
        plt.xlabel('Timestamp (second)')
        plt.ylabel('Learning Accuracy')
        plt.title(u"The scatter diagram of two classes")
        plt.show()

# now use on-the-fly analysis, and analysis the accuracy
def on_the_fly():
    sensor_emulator = Sensor_Emulator()
    sensor_emulator.delete_content("temp")
    dataset_dump_url = "http://localhost:1880/dump_data_set" # this is to set dump url
    dataset_write_temp_url = "http://localhost:1880/write_temp" # this is to set write temp url
    sensor_emulator.send_command_to_server(dataset_dump_url)
    sensor_emulator.send_command_to_server(dataset_write_temp_url)
    time.sleep(20)
    time_stamp = []
    accuracy_rate = []
    for i in range(40):
        time_stamp.append(i)
        score = sensor_emulator.analysis_and_score()
        accuracy_rate.append(score)
        print("Plot the diagram between the timestamp and accuracy")
        sensor_emulator.plot_scatter(time_stamp, accuracy_rate) # draw the diagram on the fly
        time.sleep(5)

if __name__ == '__main__':
    on_the_fly()





