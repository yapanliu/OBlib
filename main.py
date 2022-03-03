import numpy as np
import matplotlib as plt
import csv
import pandas as pd

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
occupancy = Occupancy.AshraeProfile()
haldi_lr = Windows.LogisticRegression_Haldi()
rf = Windows.RandomForest_E3D()
#haldi_lr = Windows.LogisticRegression_default()
print('windows imported')

# features from the data set
feature_strings = ['Date_Time','Indoor_Temp [C]','Outdoor_Temp [C]']
target_string = ['Windor_Status']

# features for the random forest model (svms have same features)
feature_strings = ['Indoor_Temp [C]','Outdoor_Temp [C]', 'BLINDSState_rightNow',
                   'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber' ]
target_string = ['Windor_Status']


AbsoluteMetrices = AbsoluteMetrices()

def test_model(path, file_name,file_target, feature_strings, target_string):
    '''

    '''
    data = load_data.open_file(path + file_name)  # read input features from a csv
    target = load_data.open_file(path + file_target)  # read targets from a csv

    x_test = load_data.create_x(data, feature_strings)  # get variable that contains features
    y_true = load_data.create_x(target, target_string)  # get variable that contains ground truth

    #there is still an issue with the frankfurt data set- not the same number of data points
    #in all files
    x_test = x_test[:y_true.shape[0]]
    #y_pred = haldi_lr.predict(x_test[['Indoor_CO2[ppm]']])  # get modeled OB target values
    y_pred = haldi_lr.predict(x_test[['Indoor_Temp [C]', 'Outdoor_Temp [C]']])
    model = rf.train(x_test.iloc[0:29, :], y_true.iloc[0:29])
    y_pred = rf.predict(model, x_test.iloc[0:29])

    # train random forest classifier
    
    print(y_pred)

    acc, conf_mat, f_1 = AbsoluteMetrices.occupancy(y_true.iloc[0:29], y_pred)
    print('accuracy:', acc)
    print('confusion matrix:', conf_mat)
    # tba: evaluate predictions



test_model(path, file_name,file_target, feature_strings, target_string)






