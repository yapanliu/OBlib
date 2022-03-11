import numpy as np
import matplotlib as plt
import csv
import pandas as pd

from OB.Windows import RandomForest_Aachen

from OBlib.OB import Occupancy
from OBlib.load_data import load_data
from OBlib.evaluation import AbsoluteMetrices
from OBlib.evaluation import Metrices
from OBlib.OB import Windows

from sklearn.linear_model import LogisticRegression


# path and file names
path = 'c:/git/data/'
file_name = 'Study_26_Study26.csv'
file_target = 'Study_26_Study26.csv'

load_data = load_data()

# pick up a model
# examples
# window opening model using logistic regression
haldi_lr = Windows.LogisticRegression_Haldi()

# window opening model using random forest
rf_model = RandomForest_Aachen.Model()
rf_data = RandomForest_Aachen.PrepareData()

feature_strings = rf_data.FeatureNames.feature_strings
target_string = rf_data.FeatureNames.target_string


AbsoluteMetrices = AbsoluteMetrices()

def test_model(path, file_name,file_target, feature_strings, target_string):
    '''

    '''
    data = load_data.open_file(path + file_name)  # read input features from a csv

    data = load_data.create_x(data, rf_data.FeatureNames.feature_strings)

    target = load_data.open_file(path + file_target)  # read targets from a csv

    #y = load_data.create_x(target, target_string)  # get variable that contains ground truth
    x_train, y_true, x_test, y_test = rf_data.preprocess_data(data)


    #there is still an issue with the frankfurt data set- not the same number of data points
    #model = haldi_lr.train(x_train[['Indoor_Temp [C]', 'Outdoor_Temp [C]']], y_true)
    #y_pred = haldi_lr.test(model, x_test[['Indoor_Temp [C]', 'Outdoor_Temp [C]']])
    model = rf_model.train(x_train, y_true)
    y_pred = rf_model.test(model, x_test)

    # train random forest classifier
    
    print(y_pred)

    acc, conf_mat, f_1 = AbsoluteMetrices.windows(y_test, y_pred)
    print('accuracy:', acc)
    print('confusion matrix:', conf_mat)

test_model(path, file_name,file_target, feature_strings, target_string)






