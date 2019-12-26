    # This project performs several different machine learning algorithms on the same data set and compares them.
    #
    # Nonlinear Algorithms:
    #
    #     k-Nearest Neighbors
    #     Classification and Regression Tree
    #     Support Vector Machine
    #     Naive Bayes
    #
    # Ensemble Algorithms:
    #
    #     Bagged Decision Trees
    #     Random Forest
    #     Extra Trees
    #     Gradient Boosting Machine


import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.naive_bayes import GaussianNB # naive bayes
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from numpy import concatenate
from math import sqrt
from numpy import mean
from numpy import std
from keras.layers import Dropout
from keras import regularizers
import talos as talos
import seaborn as sns
from sklearn.model_selection import GridSearchCV
# from tsfresh import extract_features
import warnings
import sklearn.exceptions
import autokeras as ak
from numpy import array
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from talos.model.normalizers import lr_normalizer
from sklearn.decomposition import PCA
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from keras import metrics
from keras import backend as K
from keras import optimizers
import time
from keras.callbacks import TensorBoard
import tikzplotlib
import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf


class Machine_Learn_Static(object):
    def __init__(self):
        self.regressor = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=5000)   # build the linear regression model
        self.gnb = GaussianNB()  # using sklearn gaussian naive bayes
        self.dt = tree.DecisionTreeClassifier()  # using sklearn decision tree
        self.svc = SVC(C=1.0, decision_function_shape='ovr', degree=3, gamma='auto_deprecated', kernel='rbf')
        self.knn = KNeighborsClassifier(n_neighbors=5)  # using sklearn k-nearest neighbors
        self.rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)   # using sklearn random forest
        self.b = BaggingClassifier()   # using sklearn bagging classification
        self.et = ExtraTreesClassifier()  # using extra tree classification
        self.gb = GradientBoostingClassifier()   # using gradient boosting classification
        self.ada = AdaBoostClassifier()  # using ada boost
        self.nn = MLPClassifier(activation='relu', early_stopping=True, hidden_layer_sizes=(5,5), max_iter=500,
                                shuffle=False, solver='sgd', validation_fraction=0.2,
                                batch_size=5, learning_rate='adaptive', learning_rate_init=0.0001)   # using Multilayer Perceptron





    # use sklearn logic regression / example program
    def logic_regress_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use sklearn to train dataset
        start = time.time()
        model = self.regressor.fit(x_train, y_train)
        stop = time.time()
        print(f"Logistic Regression training time: {stop - start}s")
        predict = self.regressor.predict(x_test)
        prob = self.regressor.predict_proba(x_test)
        # print(prob)
        # proba = pd.DataFrame(prob)
        # proba.to_csv("probabilities.csv", sep=';', encoding='utf-8', index=False)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Logistic Regression", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        # print(y_test)
        # print("predicted: ")
        # print(predict)
        self.plot_confusion_matrix("Logistic Regression", y_test, predict, classes=class_names)
        return metrics

    # # use sklearn naive bayes / example program
    def naive_bayes_regress_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use sklearn to train dataset
        start = time.time()
        model = self.gnb.fit(x_train, y_train)
        stop = time.time()
        print(f"Naive Bayes training time: {stop - start}s")
        predict = self.gnb.predict(x_test)
        # prob = self.regressor.predict_proba(x_test)
        # print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Naive Bayes", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Naive Bayes", y_test, predict, classes=class_names)
        return metrics

    # # use decision tree
    def decision_tree_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use decision tree to train dataset
        start = time.time()
        model = self.dt.fit(x_train, y_train)
        stop = time.time()
        print(f"Decision tree training time: {stop - start}s")
        predict = self.dt.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Decision Tree", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Decision Tree", y_test, predict, classes=class_names)
        feature_importances = pd.DataFrame(model.feature_importances_, index = x_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        print("Feature importances Decision Tree:")
        print(feature_importances)
        return metrics

    # # use support vector
    def support_vector_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use support vector to train dataset
        start = time.time()
        model = self.svc.fit(x_train, y_train)
        stop = time.time()
        print(f"Support Vector Machine training time: {stop - start}s")
        predict = self.svc.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Support Vector Machine", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Support Vector Machine", y_test, predict, classes=class_names)
        return metrics

    # # use k-nearest neighbors
    def k_nearest_neighbors_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use k-nearest neighbors to train dataset
        start = time.time()
        model = self.knn.fit(x_train, y_train)
        stop = time.time()
        print(f"K-Nearest Neighbors training time: {stop - start}s")
        predict = self.knn.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "K-Nearest Neighbors", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("K-Nearest Neighbors", y_test, predict, classes=class_names)
        return metrics

    def random_forest_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use random forest to train dataset
        start = time.time()
        model = self.rf.fit(x_train, y_train)
        stop = time.time()
        print(f"Random Forest training time: {stop - start}s")
        predict = self.rf.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Random Forest", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Random Forest", y_test, predict, classes=class_names)
        feature_importances = pd.DataFrame(model.feature_importances_, index = x_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        print("Feature importances Random Forest:")
        print(feature_importances)
        return metrics

    # # use bagging
    def bagging_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use bagging to train dataset
        start = time.time()
        model = self.b.fit(x_train, y_train)
        stop = time.time()
        print(f"Bagging training time: {stop - start}s")
        predict = self.b.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Bagging", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Bagging", y_test, predict, classes=class_names)
        return metrics

    # # use extra tree
    def extra_tree_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use extra tree to train dataset
        start = time.time()
        model = self.et.fit(x_train, y_train)
        stop = time.time()
        print(f"Extra Tree training time: {stop - start}s")
        predict = self.et.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Extra Tree", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Extra Tree", y_test, predict, classes=class_names)
        feature_importances = pd.DataFrame(model.feature_importances_, index = x_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        print("Feature importances Extra Forest:")
        print(feature_importances)
        return metrics

    # # use ada boost
    def ada_boost_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use ada boost to train dataset
        start = time.time()
        model = self.et.fit(x_train, y_train)
        stop = time.time()
        print(f"Ada Boost training time: {stop - start}s")
        predict = self.et.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Ada Boost", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Ada Boost", y_test, predict, classes=class_names)
        feature_importances = pd.DataFrame(model.feature_importances_, index = x_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        print("Feature importances Extra Forest:")
        print(feature_importances)
        return metrics

    # # use gradient boosting
    def gradient_boosting_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use gradient boosting to train dataset
        start = time.time()
        model = self.gb.fit(x_train, y_train)
        stop = time.time()
        print(f"Gradient Boosting training time: {stop - start}s")
        predict = self.gb.predict(x_test)
        #prob = self.regressor.predict_proba(x_test)
        #print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Gradient Boosting", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Gradient Boosting", y_test, predict, classes=class_names)
        feature_importances = pd.DataFrame(model.feature_importances_, index = x_train.columns, columns=['importance']).sort_values('importance', ascending=False)
        print("Feature importances Gradient Boosting:")
        print(feature_importances)
        return metrics

     # # use Multilayer Perceptron
    def neural_network_fit(self, x_train, y_train, x_test, y_test, metrics):
        # Now use gradient boosting to train dataset
        start = time.time()
        model = self.gb.fit(x_train, y_train)
        stop = time.time()
        print(f"Multilayer Perceptron training time: {stop - start}s")
        predict = self.gb.predict(x_test)
        # prob = self.regressor.predict_proba(x_test)
        # print(prob)
        metrics = self.print_metrics(model, y_test, x_test, predict, "Multilayer Perceptron", metrics)
        # printing the confusion matrix
        class_names = unique_labels(y_train)
        # Plot non-normalized confusion matrix
        self.plot_confusion_matrix("Multilayer Perceptron", y_test, predict, classes=class_names)
        return metrics

    # Auto Keras : https://autokeras.com/temp/supervised/
    def auto_keras(self, x_train, y_train, x_test, y_test):
        clf = ak.ImageClassifier()
        x_train = x_train.values.reshape(x_train.shape + (1,))
        x_test = x_test.values.reshape(x_test.shape + (1,))

        clf.fit(x_train, y_train)
        results = clf.predict(x_test)
        clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
        y = clf.evaluate(x_test, y_test)
        print("Auto Keras accuracy: ", y)



    # print metrics
    def print_metrics(self, model, y_test, x_test, y_pred, algorithm, metrics):
        # training accuracy
        training_accuracy = round(model.score(x_test, y_test), 2)
        print(algorithm, ": training accuracy is: ", training_accuracy)
        # testing accuracy
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        recall = round(recall_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred)), 4)
        precision = round(precision_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred)), 4)
        f1 = round(f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred)), 4)
        mcc = round(matthews_corrcoef(y_test, y_pred), 4)
        rmse = round(sqrt(mean_squared_error(y_test, y_pred)), 4)  # Mean squared error regression loss
        print("Missing classes in prediction: ", set(y_test) - set(y_pred))
        print(algorithm, ": testing accuracy is: ", accuracy, ", recall (weighted) is: ", recall,
              "precision (weighted) is: ", precision, "F1 (weighted) is: ", f1, "MCC is:", mcc, "RMSE is:", rmse)
        metric = pd.DataFrame({"algorithm": [algorithm], "accuracy": [accuracy], "recall": [recall],
                               "precision": [precision], "f1": [f1], "mcc": [mcc], "rmse": [rmse]})
        metrics = metrics.append(metric)
        print("-------------------------------------------------------------------------------------------------------")
        return metrics

    # plot metrics
    def plot_metrics(self, metrics):
        # print testing accuracy, precision and recall for each activity as bar chart
        fig = plt.figure()
        # set width of bar
        barWidth = 0.1
        # set height of bar
        accuracy = metrics["accuracy"]
        recall = metrics["recall"]
        precision = metrics["precision"]
        f1 = metrics["f1"]
        mcc = metrics["mcc"]

        # Set position of bar on X axis
        r1 = np.arange(len(accuracy))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        r4 = [x + barWidth for x in r3]
        r5 = [x + barWidth for x in r4]

        # Make the plot
        plt.bar(r1, accuracy, color='red', width=barWidth, edgecolor='white', label='Accuracy')
        plt.bar(r2, recall, color='blue', width=barWidth, edgecolor='white', label='Recall')
        plt.bar(r3, precision, color='orange', width=barWidth, edgecolor='white', label='Precision')
        plt.bar(r4, f1, color='green', width=barWidth, edgecolor='white', label='F1')
        plt.bar(r5, mcc, color='pink', width=barWidth, edgecolor='white', label='MCC')
        # Add xticks on the middle of the group bars
        # plt.xlabel('Algorithms', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(accuracy))], metrics["algorithm"], rotation=45, ha="right")
        plt.title("Metrics for different Machine Learning Algorithms")
        # Create legend and save graphic
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig('figures/metrics.png')
        fig.savefig('figures/metrics.pgf')

        fig = plt.figure()
        rmse = metrics["rmse"]
        label = metrics["algorithm"]
        index = np.arange(len(label))
        plt.bar(index, rmse)
        # plt.xlabel('Genre', fontsize=5)
        # plt.ylabel('No of Movies', fontsize=5)
        plt.xticks(index, label, rotation=45, ha="right")
        plt.title('RMSE  for different Machine Learning Algorithms')
        fig.tight_layout()
        fig.savefig('figures/rmse.png')
        fig.savefig('figures/rmse.pgf')



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




    def plot_confusion_matrix(self, algorithm, y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
        #    This function prints and plots the confusion matrix.


        np.set_printoptions(precision=2)
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        if not title:
            title = 'Confusion matrix, using ' + algorithm

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data

        # when using "no activity" as addictional class
        # classes_new = pd.DataFrame(classes[unique_labels(y_true, y_pred)], columns=["ACTIVITY"])
        classes_new = pd.DataFrame(classes[unique_labels(y_true, y_pred) - 1], columns=["ACTIVITY"])

        activity_map = {1: "Act01", 2: "Act02", 3: "Act03", 4: "Act04", 5: "Act05", 6: "Act06", 7: "Act07", 8: "Act08",
                        9: "Act09",  10: "Act10", 11: "Act11", 12: "Act12", 13: "Act13", 14: "Act14",  15: "Act15",
                        16: "Act16", 17: "Act17", 18: "Act18", 19: "Act19", 20: "Act20", 21: "Act21", 22: "Act22",
                        23: "Act23", 24: "Act24", 0: "no activity"}
        classes_new["ACTIVITY"] = classes_new["ACTIVITY"].map(activity_map)
        activities = pd.read_csv("activities.csv", delimiter=",")
        activities = activities.append(pd.DataFrame({"ACTIVITY": ["no activity"], "Activity Name": ["no activity"]}))
        classes_merged = pd.merge(classes_new, activities, on="ACTIVITY")
        labels = classes_merged["Activity Name"]


        # print(cm)

        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        ax.title.set_fontsize(16)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        fig.savefig("figures/confusion-matrix-" + algorithm + ".png")
        fig.savefig("figures/confusion-matrix-" + algorithm + ".pgf")
        return ax

    def evaluate_CNN_model_talos(self, train_X, train_y, x_val, y_val, params):

        n_output = 24  # number of classes

        model = Sequential()
        model.add(Conv1D(params['first_neuron'], kernel_size=2, activation=params['activation'], input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(MaxPooling1D(params['pool_size']))
        model.add(Flatten())
        model.add(Dense(params['second_neuron'], activation='relu'))
        model.add(Dropout(params['dropout']))
        model.add(Dense(units=n_output, activation='softmax'))
        model.compile(loss=params['losses'], optimizer='adam', metrics=['acc'])
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)


        print(model.summary())


        # fit network
        out = model.fit(train_X, train_y, epochs=50, batch_size=5, verbose=2, shuffle=False, validation_data=[x_val, y_val], callbacks=[es])
        return out, model

    # https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
    def evaluate_CNN_LSTM_model(self, train_X, train_y, test_X, test_y):


        n_outputs = 24  # number of classes

        # https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None, 32, 83))) # input_shape=(None,32,83)))
        # model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.8)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dropout(0.8))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dense(n_outputs, activation='softmax'))

        # setting up TensorBoard
        tensorboard = TensorBoard(log_dir="logs/cnn-lstm/{}".format(time()))

        optimizer = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # fit network
        history = model.fit(train_X, train_y, epochs=50, batch_size=5, verbose=2, shuffle=False, validation_split=0.2, callbacks=[es, tensorboard])
        print(model.summary())

        pyplot.clf()
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.plot(history.history['acc'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        #pyplot.show()

        # plot metrics
        #pyplot.clf()
        #pyplot.plot(history.history['acc'])
        #pyplot.show()


        #print("metrics are ", model.metrics_names)
        loss, accuracy = model.evaluate(test_X, test_y, verbose=2)

        yhat = model.predict(test_X, verbose=1)
        y_pred = model.predict_classes(test_X)

        # reverse one hot encoding
        predictions = pd.DataFrame(y_pred)
        trues = pd.DataFrame(test_y)

        ground_truth = trues.idxmax(1).values
        class_names = unique_labels(train_y)

        # add 1 to make sure the classes are mapped correctly with labels
        ground_truth = ground_truth + 1
        predictions += 1
        # class_names = unique_labels(ground_truth)
        class_names = unique_labels(list(range(1, 25)))

        self.plot_confusion_matrix("CNN with LSTM", ground_truth, predictions, classes=class_names)

        recall = round(recall_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        precision = round(precision_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        f1 = round(f1_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        mcc = round(matthews_corrcoef(ground_truth, predictions), 4)

        # print metrics
        print(history.history['loss'])
        print(history.history['acc'])
        print(history.history['val_loss'])
        print(history.history['val_acc'])
        print("recall", recall)
        print("precision", precision)
        print("f1", f1)
        print("mcc", mcc)

        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        print('RMSE: %.3f' % rmse)


        print("classification report:")
        print(classification_report(ground_truth, predictions))
        return loss, accuracy, rmse, recall, precision, f1, mcc



    # fit and evaluate a CNN model
    def evaluate_CNN_model(self, train_X, train_y, test_X, test_y):
        # https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        n_outputs = 24  # number of classes
        epochs = 50

        bn = BatchNormalization()
        # batch normalisation: https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
        # learning rate: https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/

        model = Sequential()
        # model.add(BatchNormalization(input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]))) # , kernel_regularizer=regularizers.l2(0.0001)
        # model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        # model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu')) # , kernel_regularizer=regularizers.l2(0.0001)
        #model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(n_outputs, activation='softmax'))

        # setting up TensorBoard
        tensorboard = TensorBoard(log_dir="logs/cnn/{}".format(time.time()))

        # optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0) # no bueno
        # optimizer = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
        # optimizer = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        # decay = 0.0001/epochs   # lr/epochs    # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
        # optimizer = optimizers.SGD(lr=0.0001, momentum=0.8, decay=decay, nesterov=False)
        # optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # 0.001 as lr is default
        optimizer = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004) # 67%
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc']) # ,f1_m,precision_m, recall_m
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # ,f1_m,precision_m, recall_m

        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        # fit network
        start = time.time()
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=5, verbose=2, shuffle=False, validation_split=0.25, callbacks=[es, tensorboard])
        stop = time.time()
        print(f"CNN training time: {stop - start}s")
        print(model.summary())

        pyplot.clf()
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.plot(history.history['acc'])
        pyplot.title('CNN: model train vs validation loss and accuracy')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation', 'accuracy'], loc='upper right')
        # pyplot.show()

        # plot metrics
        #pyplot.clf()
        #pyplot.plot(history.history['acc'])
        #pyplot.show()


        #print("metrics are ", model.metrics_names)
        loss, accuracy = model.evaluate(test_X, test_y, verbose=2)

        yhat = model.predict(test_X, verbose=1)
        y_pred = model.predict_classes(test_X)

        # reverse one hot encoding
        predictions = pd.DataFrame(y_pred)
        trues = pd.DataFrame(test_y)

        ground_truth = trues.idxmax(1).values
        class_names = unique_labels(train_y)

        # add 1 to make sure the classes are mapped correctly with labels
        ground_truth = ground_truth + 1
        predictions += 1
        # class_names = unique_labels(ground_truth)
        class_names = unique_labels(list(range(1, 25)))

        self.plot_confusion_matrix("CNN", ground_truth, predictions, classes=class_names)

        recall = round(recall_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        precision = round(precision_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        f1 = round(f1_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        mcc = round(matthews_corrcoef(ground_truth, predictions), 4)

        # print metrics
        print(history.history['loss'])
        print(history.history['acc'])
        print(history.history['val_loss'])
        print(history.history['val_acc'])
        print("recall", recall)
        print("precision", precision)
        print("f1", f1)
        print("mcc", mcc)



        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        print('RMSE: %.3f' % rmse)


        print("classification report:")
        print(classification_report(ground_truth, predictions))
        return loss, accuracy, rmse, recall, precision, f1, mcc


    # fit and evaluate a LSTM model
    def evaluate_LSTM_model(self, train_X, train_y, test_X, test_y):
        # design network

        # compute hidden nodes
        # https://towardsdatascience.com/choosing-the-right-hyperparameters-for-a-simple-lstm-using-keras-f8e9ed76f046
        n_samples = train_X.shape[0]
        n_outputs = 24  # number of classes
        # n_input = 69  # number of features
        n_input = 83  # number of features
        alpha = 2  # 2-10
        n_hidden = int(n_samples / (alpha * (n_input + n_outputs)))

        # print("hidden neurons are ", n_hidden)

        model = Sequential()
        # https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e
        model.add(LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_regularizer=regularizers.l2(0.0001)))
        # model.add(Dropout(0.2))  # https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/
        # model.add(BatchNormalization())

        model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
        model.add(Dense(n_outputs, activation='softmax'))
        # Note that we use a “softmax” activation function in the output layer. This is to ensure the output values are
        # in the range of 0 and 1 and may be used as predicted probabilities.

        # setting up TensorBoard
        tensorboard = TensorBoard(log_dir="logs/lstm/{}".format(time.time()))

        optimizer = optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc']) # ,f1_m,precision_m, recall_m

        # early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)



        # fit network
        start  = time.time()
        history = model.fit(train_X, train_y, epochs=50, batch_size=4, verbose=2, shuffle=False, validation_split=0.2, callbacks=[es, tensorboard])
        stop = time.time()
        print(f"LSTM training time: {stop - start}s")
        print(model.summary())


        pyplot.clf()
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.plot(history.history['acc'])
        pyplot.title('LSTM: model train vs validation loss and accuracy')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation', 'accuracy'], loc='upper right')
        # pyplot.show()

        print(history.history['loss'])
        print(history.history['acc'])
        print(history.history['val_loss'])
        print(history.history['val_acc'])

        #print("metrics are ", model.metrics_names)
        loss, accuracy = model.evaluate(test_X, test_y, verbose=2)

        yhat = model.predict(test_X, verbose=1)
        y_pred = model.predict_classes(test_X)

        # reverse one hot encoding
        predictions = pd.DataFrame(y_pred)
        trues = pd.DataFrame(test_y)

        ground_truth = trues.idxmax(1).values
        class_names = unique_labels(train_y)

        # add 1 to make sure the classes are mapped correctly with labels
        ground_truth = ground_truth + 1
        predictions += 1
        # class_names = unique_labels(ground_truth)
        class_names = unique_labels(list(range(1, 25)))

        self.plot_confusion_matrix("LSTM", ground_truth, predictions, classes=class_names)

        recall = round(recall_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        precision = round(precision_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        f1 = round(f1_score(ground_truth, predictions, average="weighted", labels=np.unique(predictions)), 4)
        mcc = round(matthews_corrcoef(ground_truth, predictions), 4)

        # print metrics
        print(history.history['loss'])
        print(history.history['acc'])
        print(history.history['val_loss'])
        print(history.history['val_acc'])
        print("recall", recall)
        print("precision", precision)
        print("f1", f1)
        print("mcc", mcc)

        rmse = sqrt(mean_squared_error(ground_truth, predictions))
        print('RMSE: %.3f' % rmse)


        print("classification report:")
        print(classification_report(ground_truth, predictions))
        return loss, accuracy, rmse, recall, precision, f1, mcc



    # summarize scores
    def summarize_results(self, accuracies, losses, recalls, precisions, f1s, mccs, rmses, algorithm, metrics):
        print(accuracies)
        print(losses)
        a_m, a_s = mean(accuracies), std(accuracies)
        l_m, l_s = mean(losses), std(losses)
        r_m, r_s = mean(recalls), std(recalls)
        p_m, p_s = mean(precisions), std(precisions)
        f_m, f_s = mean(f1s), std(f1s)
        m_m, m_s = mean(mccs), std(mccs)
        rm_m, rm_s = mean(rmses), std(rmses)
        print('Accuracy: %.3f%% (+/-%.3f)' % (a_m, a_s))
        print('Loss: %.3f%% (+/-%.3f)' % (l_m, l_s))
        print('Recall: %.3f%% (+/-%.3f)' % (r_m, r_s))
        print('Precision: %.3f%% (+/-%.3f)' % (p_m, p_s))
        print('F1: %.3f%% (+/-%.3f)' % (f_m, f_s))
        print('MCC: %.3f%% (+/-%.3f)' % (m_m, m_s))
        print('RMSE: %.3f%% (+/-%.3f)' % (rm_m, rm_s))
        metric = pd.DataFrame({"algorithm": [algorithm], "accuracy": [a_m], "recall": [r_m],
                               "precision": [p_m], "f1": [f_m], "mcc": [m_m], "rmse": [rm_m]})
        metrics = metrics.append(metric)
        return metrics


def HAR_classification():
    # read a dataset

    classification = Machine_Learn_Static()

    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": [
             r"\usepackage[utf8x]{inputenc}",
             r"\usepackage[T1]{fontenc}",
             r"\usepackage{cmbright}",
             ]
    })
    training_samples = pd.read_csv("samples-training.csv",  delimiter=";")
    test_samples = pd.read_csv("samples-testing.csv",  delimiter=";")

    # eliminate "no activity" samples
    training_samples = training_samples[training_samples["activity"] != 0]
    test_samples = test_samples[test_samples["activity"] != 0]


    test_samples = test_samples.drop(["TIMESTAMP"], axis=1)
    test_samples = test_samples.astype(float)

    training_samples = training_samples.drop(["TIMESTAMP"], axis=1)
    training_samples = training_samples.astype(float)




    # prepare for ML algorithms

    timesteps = 40
    n_features = training_samples.shape[1] - 1
    use_lag_features = True

    if(use_lag_features):


        reframed_test = classification.series_to_supervised(test_samples, timesteps, 1)
        reframed_training = classification.series_to_supervised(training_samples, timesteps, 1)


        for i in range(timesteps):
            reframed_test = reframed_test.drop(["var" + str(n_features +1) + "(t-" + str(i + 1) + ")"], axis=1)  # getting rid of class label
            reframed_training = reframed_training.drop(["var" + str(n_features +1) + "(t-" + str(i + 1) + ")"], axis=1)  # getting rid of class label

        x_train, y_train = reframed_training.iloc[:, :-1], reframed_training.iloc[:, -1]
        x_test, y_test = reframed_test.iloc[:, :-1], reframed_test.iloc[:, -1]


        # print(x_train)
        # print(y_train)

    else:

        x_train, y_train = training_samples.iloc[:, :-1], training_samples.iloc[:, -1]
        x_test, y_test = test_samples.iloc[:, :-1], test_samples.iloc[:, -1]


    print("Running ML algorithms with the following parameters:")
    print("Lags (time steps): " + str(timesteps))
    print("Features: " + str(x_train.shape[1]))
    print("Training samples: " + str(len(x_train)))
    print("Testing samples: " + str(len(x_test)))




############# standard machine learning algorithms #####################################################################


    # run machine learning algorithms - uncomment this if you want to run
    metrics = pd.DataFrame(columns=["algorithm", "accuracy", "recall", "precision", "f1", "mcc", "rmse"])

    # Logistic Regression
    metrics = classification.logic_regress_fit(x_train, y_train, x_test, y_test, metrics)

    # Naive Bayes
    metrics = classification.naive_bayes_regress_fit(x_train, y_train, x_test, y_test, metrics)

    # Desicion Tree
    metrics = classification.decision_tree_fit(x_train, y_train, x_test, y_test, metrics)

    # Support Vector Classification
    metrics = classification.support_vector_fit(x_train, y_train, x_test, y_test, metrics)

    # k-nearest neighbors
    metrics = classification.k_nearest_neighbors_fit(x_train, y_train, x_test, y_test, metrics)

    # random forest
    metrics = classification.random_forest_fit(x_train, y_train, x_test, y_test, metrics)

    # bagging
    metrics = classification.bagging_fit(x_train, y_train, x_test, y_test, metrics)

    # extra tree
    metrics = classification.extra_tree_fit(x_train, y_train, x_test, y_test, metrics)

    # gradient boosting
    metrics = classification.gradient_boosting_fit(x_train, y_train, x_test, y_test, metrics)

    # Multilayer Perceptron
    metrics = classification.neural_network_fit(x_train, y_train, x_test, y_test, metrics)


    print("Summary of results:")
    print(metrics)
    classification.plot_metrics(metrics)






    # Auto Keras : https://autokeras.com/temp/supervised/ - uncomment if you want to run AutoKeras (takes a couple of
    # days to finish
    # classification.auto_keras(x_train, y_train, x_test, y_test)


############# deep learning algorithms #################################################################################


    # Keras LSTM: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
    # https://machinelearningmastery.com/time-series-forecasting-supervised-learning/

    # reshape input to be 3D [samples, timesteps, features per step]
    #print(x_train.shape)
    train_X = x_train.values.reshape(x_train.shape[0], timesteps + 1, n_features)
    test_X = x_test.values.reshape(x_test.shape[0], timesteps + 1, n_features)




    # make classes start with class 0 instead of class 1
    y_test = y_test - 1
    y_train = y_train - 1

    train_Y = to_categorical(y_train, num_classes=24)
    test_Y = to_categorical(y_test, num_classes=24)
    # print("shapes")
    # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)



    repeats = 10
    # repeat experiment
    accuracies = list()
    losses = list()
    recalls = list()
    precisions = list()
    f1s = list()
    mccs = list()
    rmses = list()

    for r in range(repeats):
        # loss, accuracy, rmse, recall, precision, f1, mcc = classification.evaluate_LSTM_model(train_X, train_Y, test_X, test_Y) # run LSTM
        loss, accuracy, rmse, recall, precision, f1, mcc = classification.evaluate_CNN_model(train_X, train_Y, test_X, test_Y) # run CNN
        # loss, accuracy, rmse, recall, precision, f1, mcc = classification.evaluate_CNN_LSTM_model(train_X, train_Y, test_X, test_Y) # run CNN with LSTM layers
        accuracy = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, accuracy))
        accuracies.append(accuracy)
        losses.append(loss)
        rmses.append(rmse)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        mccs.append(mcc)
    metrics = classification.summarize_results(accuracies, losses, recalls, precisions, f1s, mccs, rmses, "CNN", metrics)

    for r in range(repeats):
        loss, accuracy, rmse, recall, precision, f1, mcc = classification.evaluate_LSTM_model(train_X, train_Y, test_X, test_Y) # run LSTM
        #loss, accuracy, rmse, recall, precision, f1, mcc = classification.evaluate_CNN_model(train_X, train_Y, test_X, test_Y) # run CNN
        #loss, accuracy, rmse, recall, precision, f1, mcc = classification.evaluate_CNN_LSTM_model(train_X, train_Y, test_X, test_Y) # run CNN with LSTM layers
        accuracy = accuracy * 100.0
        print('>#%d: %.3f' % (r+1, accuracy))
        accuracies.append(accuracy)
        losses.append(loss)
        rmses.append(rmse)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        mccs.append(mcc)
    metrics = classification.summarize_results(accuracies, losses, recalls, precisions, f1s, mccs, rmses, "LSTM", metrics)

    classification.plot_metrics(metrics)


if __name__ == '__main__':
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)

    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    HAR_classification()

