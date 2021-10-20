import numpy as np
import matplotlib as plt
import csv
import pandas as pd

from OBlib.OB import occupancy
from OBlib.load_data import load_data
from OBlib.evaluation import AbsoluteMetrices
from OBlib.OB import windows

from sklearn.linear_model import LogisticRegression


# path and file names
path = 'c:/Obdata/data/'
file_name = 'Indoor_Measurement_Study24.csv'
file_target = 'Occupancy_Measurement_Study24.csv'

load_data = load_data()
occupancy = occupancy.occupancy()


    
# features from the data set
feature_strings = ['Date_Time','Indoor_Temp[C]','Indoor_CO2[ppm]','Room_ID','Building_ID']
target_string = ['Occupancy_Measurement[0-Unoccupied1-Occupied]']


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
    y_pred = occupancy.ashrae(x_test[['Indoor_CO2[ppm]']])  # get modeled OB target values
    # tba: evaluate predictions



test_model(path, file_name,file_target, feature_strings, target_string)






