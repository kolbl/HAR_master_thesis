# Sensor-based human activity recognition

Master thesis project of Linda Kolb, University of Technology, Graz. 

This project takes raw sensor data from the 
[UCAmI 2018 cup](https://www.mdpi.com/2504-3900/2/19/1267). The sensor 
data comes from binary sensors, an intelligent floor, acceleration data 
from a smart watch and proximity sensors. The sensors were set up in a 
smart lab. A test user was living in the smart lab for 10 days. The goal 
of this project is to classify activities of daily living. 

The captured data is split into two sets - training (7 days) and testing 
(3 days). The sensor data is preprocessed and analysed. There are several 
standard machine learning algorithms being used for classification:

* Nonlinear Algorithms:
  * Nearest Neighbors
  * Classification and Regression Tree
  * Support Vector Machine
  * Naive Bayes
* Ensemble Algorithms:
  * Bagged Decision Trees
  * Random Forest
  * Extra Trees
  * Gradient Boosting Machine


Additionally, several deep learning algorithms are being explored:
* CNN
* LSTM
* CNN with LSTM layers

In the future, an on-the-fly classification of the human activities 
will be implemented using [Node-RED](https://nodered.org/). 
 
## Prerequisites

To run this project locally, you need python, pandas, tensorflow, 
keras, numpy, matplotlib, talos and sci-kit learn.

## Directory structure

```
src/
├── activities.csv -- list of all types of activities of daily living
├── labels_training_set.csv -- timestamp, segment and activity of training set
├── documentation/  
│   └── 2019-08-31_thesis.pdf -- thesis draft
├── Data/ -- raw sensor data
│   ├── Test/ -- raw sensor data from the testing set
│   │   ├── 2017-11-09/
│   │   │   ├── 2017-11-09-A/
│   │   │   │   ├── 2017-11-09-A-acceleration.csv
│   │   │   │   ├── 2017-11-09-A-floor.csv
│   │   │   │   ├── 2017-11-09-A-proximity.csv
│   │   │   │   └── 2017-11-09-A-sensors.csv
│   │   │   ├── 2017-11-09-B/
│   │   │   │   ├── 2017-11-09-B-acceleration.csv
│   │   │   │   ├── 2017-11-09-B-floor.csv
│   │   │   │   ├── 2017-11-09-B-proximity.csv
│   │   │   │   └── 2017-11-09-B-sensors.csv
│   │   │   └── 2017-11-09-C/
│   │   │       ├── 2017-11-09-C-acceleration.csv
│   │   │       ├── 2017-11-09-C-floor.csv
│   │   │       ├── 2017-11-09-C-proximity.csv
│   │   │       └── 2017-11-09-C-sensors.csv
│   │   ├── 2017-11-13/
│   │   │   ├── 2017-11-13-A/
│   │   │   │   ├── 2017-11-13-A-acceleration.csv
│   │   │   │   ├── 2017-11-13-A-floor.csv
│   │   │   │   ├── 2017-11-13-A-proximity.csv
│   │   │   │   └── 2017-11-13-A-sensors.csv
│   │   │   ├── 2017-11-13-B/
│   │   │   │   ├── 2017-11-13-B-acceleration.csv
│   │   │   │   ├── 2017-11-13-B-floor.csv
│   │   │   │   ├── 2017-11-13-B-proximity.csv
│   │   │   │   └── 2017-11-13-B-sensors.csv
│   │   │   └── 2017-11-13-C/
│   │   │       ├── 2017-11-13-C-acceleration.csv
│   │   │       ├── 2017-11-13-C-floor.csv
│   │   │       ├── 2017-11-13-C-proximity.csv
│   │   │       └── 2017-11-13-C-sensors.csv
│   │   └── 2017-11-21/
│   │       ├── 2017-11-21-A/
│   │       │   ├── 2017-11-21-A-acceleration.csv
│   │       │   ├── 2017-11-21-A-floor.csv
│   │       │   ├── 2017-11-21-A-proximity.csv
│   │       │   ├── 2017-11-21-A-sensors.csv
│   │       │   └── _~lock.2017-11-21-A-activity.csv#
│   │       ├── 2017-11-21-B/
│   │       │   ├── 2017-11-21-B-acceleration.csv
│   │       │   ├── 2017-11-21-B-floor.csv
│   │       │   ├── 2017-11-21-B-proximity.csv
│   │       │   └── 2017-11-21-B-sensors.csv
│   │       └── 2017-11-21-C/
│   │           ├── 2017-11-21-C-acceleration.csv
│   │           ├── 2017-11-21-C-floor.csv
│   │           ├── 2017-11-21-C-proximity.csv
│   │           └── 2017-11-21-C-sensors.csv
│   └── Training/ -- raw testing data from the training set
│       ├── 2017-10-31/
│       │   ├── 2017-10-31-A/
│       │   │   ├── 2017-10-31-A-acceleration.csv
│       │   │   ├── 2017-10-31-A-activity.csv
│       │   │   ├── 2017-10-31-A-floor.csv
│       │   │   ├── 2017-10-31-A-proximity.csv
│       │   │   └── 2017-10-31-A-sensors.csv
│       │   ├── 2017-10-31-B/
│       │   │   ├── 2017-10-31-B-acceleration.csv
│       │   │   ├── 2017-10-31-B-activity.csv
│       │   │   ├── 2017-10-31-B-floor.csv
│       │   │   ├── 2017-10-31-B-proximity.csv
│       │   │   └── 2017-10-31-B-sensors.csv
│       │   └── 2017-10-31-C/
│       │       ├── 2017-10-31-C-acceleration.csv
│       │       ├── 2017-10-31-C-activity.csv
│       │       ├── 2017-10-31-C-floor.csv
│       │       ├── 2017-10-31-C-proximity.csv
│       │       └── 2017-10-31-C-sensors.csv
│       ├── 2017-11-02/
│       │   ├── 2017-11-02-A/
│       │   │   ├── 2017-11-02-A-acceleration.csv
│       │   │   ├── 2017-11-02-A-activity.csv
│       │   │   ├── 2017-11-02-A-floor.csv
│       │   │   ├── 2017-11-02-A-proximity.csv
│       │   │   └── 2017-11-02-A-sensors.csv
│       │   ├── 2017-11-02-B/
│       │   │   ├── 2017-11-02-B-acceleration.csv
│       │   │   ├── 2017-11-02-B-activity.csv
│       │   │   ├── 2017-11-02-B-floor.csv
│       │   │   ├── 2017-11-02-B-proximity.csv
│       │   │   └── 2017-11-02-B-sensors.csv
│       │   └── 2017-11-02-C/
│       │       ├── 2017-11-02-C-acceleration.csv
│       │       ├── 2017-11-02-C-activity.csv
│       │       ├── 2017-11-02-C-floor.csv
│       │       ├── 2017-11-02-C-proximity.csv
│       │       └── 2017-11-02-C-sensors.csv
│       ├── 2017-11-03/
│       │   ├── 2017-11-03-A/
│       │   │   ├── 2017-11-03-A-acceleration.csv
│       │   │   ├── 2017-11-03-A-activity.csv
│       │   │   ├── 2017-11-03-A-floor.csv
│       │   │   ├── 2017-11-03-A-proximity.csv
│       │   │   └── 2017-11-03-A-sensors.csv
│       │   ├── 2017-11-03-B/
│       │   │   ├── 2017-11-03-B-acceleration.csv
│       │   │   ├── 2017-11-03-B-activity.csv
│       │   │   ├── 2017-11-03-B-floor.csv
│       │   │   ├── 2017-11-03-B-proximity.csv
│       │   │   └── 2017-11-03-B-sensors.csv
│       │   └── 2017-11-03-C/
│       │       ├── 2017-11-03-C-acceleration.csv
│       │       ├── 2017-11-03-C-activity.csv
│       │       ├── 2017-11-03-C-floor.csv
│       │       ├── 2017-11-03-C-proximity.csv
│       │       └── 2017-11-03-C-sensors.csv
│       ├── 2017-11-08/
│       │   ├── 2017-11-08-A/
│       │   │   ├── 2017-11-08-A-acceleration.csv
│       │   │   ├── 2017-11-08-A-activity.csv
│       │   │   ├── 2017-11-08-A-floor.csv
│       │   │   ├── 2017-11-08-A-proximity.csv
│       │   │   └── 2017-11-08-A-sensors.csv
│       │   ├── 2017-11-08-B/
│       │   │   ├── 2017-11-08-B-acceleration.csv
│       │   │   ├── 2017-11-08-B-activity.csv
│       │   │   ├── 2017-11-08-B-floor.csv
│       │   │   ├── 2017-11-08-B-proximity.csv
│       │   │   └── 2017-11-08-B-sensors.csv
│       │   └── 2017-11-08-C/
│       │       ├── 2017-11-08-C-acceleration.csv
│       │       ├── 2017-11-08-C-activity.csv
│       │       ├── 2017-11-08-C-floor.csv
│       │       ├── 2017-11-08-C-proximity.csv
│       │       └── 2017-11-08-C-sensors.csv
│       ├── 2017-11-10/
│       │   ├── 2017-11-10-A/
│       │   │   ├── 2017-11-10-A-acceleration.csv
│       │   │   ├── 2017-11-10-A-activity.csv
│       │   │   ├── 2017-11-10-A-floor.csv
│       │   │   ├── 2017-11-10-A-proximity.csv
│       │   │   └── 2017-11-10-A-sensors.csv
│       │   ├── 2017-11-10-B/
│       │   │   ├── 2017-11-10-B-acceleration.csv
│       │   │   ├── 2017-11-10-B-activity.csv
│       │   │   ├── 2017-11-10-B-floor.csv
│       │   │   ├── 2017-11-10-B-proximity.csv
│       │   │   └── 2017-11-10-B-sensors.csv
│       │   └── 2017-11-10-C/
│       │       ├── 2017-11-10-C-acceleration.csv
│       │       ├── 2017-11-10-C-activity.csv
│       │       ├── 2017-11-10-C-floor.csv
│       │       ├── 2017-11-10-C-proximity.csv
│       │       └── 2017-11-10-C-sensors.csv
│       ├── 2017-11-15/
│       │   ├── 2017-11-15-A/
│       │   │   ├── 2017-11-15-A-acceleration.csv
│       │   │   ├── 2017-11-15-A-activity.csv
│       │   │   ├── 2017-11-15-A-floor.csv
│       │   │   ├── 2017-11-15-A-proximity.csv
│       │   │   └── 2017-11-15-A-sensors.csv
│       │   ├── 2017-11-15-B/
│       │   │   ├── 2017-11-15-B-acceleration.csv
│       │   │   ├── 2017-11-15-B-activity.csv
│       │   │   ├── 2017-11-15-B-floor.csv
│       │   │   ├── 2017-11-15-B-proximity.csv
│       │   │   └── 2017-11-15-B-sensors.csv
│       │   └── 2017-11-15-C/
│       │       ├── 2017-11-15-C-acceleration.csv
│       │       ├── 2017-11-15-C-activity.csv
│       │       ├── 2017-11-15-C-floor.csv
│       │       ├── 2017-11-15-C-proximity.csv
│       │       └── 2017-11-15-C-sensors.csv
│       └── 2017-11-20/
│           ├── 2017-11-20-A/
│           │   ├── 2017-11-20-A-acceleration.csv
│           │   ├── 2017-11-20-A-activity.csv
│           │   ├── 2017-11-20-A-floor.csv
│           │   ├── 2017-11-20-A-proximity.csv
│           │   └── 2017-11-20-A-sensors.csv
│           ├── 2017-11-20-B/
│           │   ├── 2017-11-20-B-acceleration.csv
│           │   ├── 2017-11-20-B-activity.csv
│           │   ├── 2017-11-20-B-floor.csv
│           │   ├── 2017-11-20-B-proximity.csv
│           │   └── 2017-11-20-B-sensors.csv
│           └── 2017-11-20-C/
│               ├── 2017-11-20-C-acceleration.csv
│               ├── 2017-11-20-C-activity.csv
│               ├── 2017-11-20-C-floor.csv
│               ├── 2017-11-20-C-proximity.csv
│               └── 2017-11-20-C-sensors.csv
├── training_data_analysis.ipynb -- exploratory data analysis of training set (Python notebook)
├── training_data_preparation.ipynb -- data preparation of training set(Python notebook)
├── figures/ -- figures for thesis
│   ├── acceleration.png
│   ├── activities-segment-a-2017-10-31.png
│   ├── all_features.png
│   ├── CNN.png
│   ├── confusion-matrix-Ada Boost.png
│   ├── confusion-matrix-Bagging.png
│   ├── confusion-matrix-CNN with LSTM.png
│   ├── confusion-matrix-CNN.png
│   ├── confusion-matrix-Decision Tree.png
│   ├── confusion-matrix-Extra Tree.png
│   ├── confusion-matrix-Gradient Boosting.png
│   ├── confusion-matrix-K-Nearest Neighbors.png
│   ├── confusion-matrix-Logistic Regression.png
│   ├── confusion-matrix-LSTM.png
│   ├── confusion-matrix-Naive Bayes.png
│   ├── confusion-matrix-Neural Network.png
│   ├── confusion-matrix-Random Forest.png
│   ├── confusion-matrix-Support Vector Machine.png
│   ├── durations.png
│   ├── frequencies-test.png
│   ├── frequencies.png
│   ├── metrics-unnormal.png
│   ├── metrics.png
│   ├── pearson-testing.png
│   ├── pearson-zeroes.png
│   ├── pearson.png
│   └── rmse.png
├── logs/ -- all kinds of logs 
│   ├── cnn/ -- TensorBoard logs of CNN
│   │   ├── 1561538510.0961337/
│   │   │   └── events.out.tfevents.1561538510.Tardis
│   │   ├── 1561538600.4727807/
│   │   │   └── events.out.tfevents.1561538600.Tardis
│   │   ├── 1561538675.765975/
│   │   │   └── events.out.tfevents.1561538676.Tardis
│   │   ├── 1561538737.6846957/
│   │   │   └── events.out.tfevents.1561538738.Tardis
│   │   ├── 1561538841.1350489/
│   │   │   └── events.out.tfevents.1561538841.Tardis
│   │   ├── 1561538925.7259188/
│   │   │   └── events.out.tfevents.1561538926.Tardis
│   │   ├── 1561538989.4388185/
│   │   │   └── events.out.tfevents.1561538990.Tardis
│   │   ├── 1561539262.2309036/
│   │   │   └── events.out.tfevents.1561539263.Tardis
│   │   ├── 1561539355.0817773/
│   │   │   └── events.out.tfevents.1561539355.Tardis
│   │   ├── 1561539442.6345506/
│   │   │   └── events.out.tfevents.1561539444.Tardis
│   │   ├── 1567245820.2177086/
│   │   │   └── events.out.tfevents.1567245820.Tardis
│   │   ├── 1567245878.8008988/
│   │   │   └── events.out.tfevents.1567245879.Tardis
│   │   ├── 1567245946.2494502/
│   │   │   └── events.out.tfevents.1567245946.Tardis
│   │   ├── 1567246007.4267726/
│   │   │   └── events.out.tfevents.1567246007.Tardis
│   │   ├── 1567246083.3909922/
│   │   │   └── events.out.tfevents.1567246083.Tardis
│   │   ├── 1567246158.3424335/
│   │   │   └── events.out.tfevents.1567246159.Tardis
│   │   ├── 1567246244.8607411/
│   │   │   └── events.out.tfevents.1567246245.Tardis
│   │   ├── 1567246302.2222462/
│   │   │   └── events.out.tfevents.1567246302.Tardis
│   │   ├── 1567246374.9865212/
│   │   │   └── events.out.tfevents.1567246375.Tardis
│   │   └── 1567246459.11666/
│   │       └── events.out.tfevents.1567246460.Tardis
│   ├── CNN_output.txt
│   ├── HAR_1.csv
│   ├── HAR_2.csv
│   ├── HAR_2.xlsx
│   ├── HAR_2_result.txt
│   ├── lstm/  -- TensorBoard logs of LSTM
│   │   ├── 1561542397.0298202/
│   │   │   └── events.out.tfevents.1561542398.Tardis
│   │   ├── 1561543285.8465974/
│   │   │   └── events.out.tfevents.1561543287.Tardis
│   │   ├── 1561544464.317077/
│   │   │   └── events.out.tfevents.1561544465.Tardis
│   │   ├── 1561545266.7511342/
│   │   │   └── events.out.tfevents.1561545268.Tardis
│   │   ├── 1561546001.9186914/
│   │   │   └── events.out.tfevents.1561546003.Tardis
│   │   ├── 1561546864.9162202/
│   │   │   └── events.out.tfevents.1561546866.Tardis
│   │   ├── 1561547907.9087794/
│   │   │   └── events.out.tfevents.1561547910.Tardis
│   │   ├── 1561548580.5796158/
│   │   │   └── events.out.tfevents.1561548582.Tardis
│   │   ├── 1561549647.4216294/
│   │   │   └── events.out.tfevents.1561549649.Tardis
│   │   ├── 1561550682.0635397/
│   │   │   └── events.out.tfevents.1561550683.Tardis
│   │   └── 1567243044.7528172/
│   │       └── events.out.tfevents.1567243046.Tardis
│   ├── LSTM_output.txt
│   └── talos_result.txt
├── samples-testing.csv -- preprocessed samples for testing
├── samples-training.csv -- preprocessed samples for training
├── HAR_classification.py -- classification algorithms (Python script) 
├── labels_testing_set.csv -- timestamp, segment and activities for test set
├── testing_data_analysis.ipynb -- exploratory data analysis of testing set (Python notebook)
└── testing_data_preparation.ipynb -- data preparation of testing set(Python notebook)
```


## How to run

Python scripts or Python notebooks are executed using the PyCharm IDE.
Shift+F10 runs all cells in the IDE. 

Notebooks can also be executed using using [Google Colab](https://colab.research.google.com) 
which offers free GPU computation. 

The scripts and notebooks can be run independently. In case the raw data 
changes, run the data preparation notebooks first and then run the 
classification. The data preparation creates the samples for testing and 
training and saves them to a CSV file which serves as input for the 
classification. 

It is possible to use [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
to visualise the learning of the deep learning algorithms. 

Start TensorBoard for CNN and LSTM logs:
```
tensorboard --logdir=src/logs/
```

 
