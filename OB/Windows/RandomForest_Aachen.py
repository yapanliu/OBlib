from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio
import tensorflow as tf
import logging
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class Study26():
    '''
    This class contains functions for preprocessing data from the study 26 from
    ashrae ob database for the random forest model

    '''

    class FeatureNames:
        '''
        naming of the required features for the random forest model in the
        study 26 from the ashrae ob database
        
        remarks:
        'BLINDSState_rightNow' is removed, since only 30 points are recorded
        '''
        feature_strings = ['Date_Time', 'Indoor_Temp [C]', 'Outdoor_Temp [C]',
                           'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']
        target_string = ['Windor_Status']

    def preprocess_data(self, x):
        '''
        use this function to do all preprocessing related to the used method
        '''
        dates = ['01-Jul', '02-Jul', '03-Jul', '04-Jul', '05-Jul', '06-Jul', '07-Jul',
                 '01-Jan', '02-Jan', '03-Jan', '04-Jan', '05-Jan', '06-Jan', '07-Jan',
                 '01-Apr', '02-Apr', '03-Apr', '04-Apr', '05-Apr', '06-Apr', '07-Apr',
                 '01-Oct', '02-Oct', '03-Oct', '04-Oct', '05-Oct', '06-Oct', '07-Oct']

        for current_date in dates:
            if current_date == '01-Jul':
                df = x[x['Date_Time'].str.contains(current_date)]
                df_test = x[x['Date_Time'].str.contains(current_date) == False]

            else:
                df_ = x[x['Date_Time'].str.contains(current_date)]
                df_test = df_test[df_test['Date_Time'].str.contains(current_date) == False]
                df = pd.concat([df, df_])

        x_train = df[
            ['Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']]
        y_true = df[['Windor_Status']]

        x_test = df_test[
            ['Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']]
        y_test = df_test[['Windor_Status']]

        return x_train, y_true, x_test, y_test




class Model():
    '''
    toDo: add model description
    '''

    class hyperparameters:
        '''
        optimal hyperparameters as presented in the paper
        '''
        trees = 100
        depth = 2


    def train(self, X, y):
        '''
        train a random forest classifier with the hyperparameters as defined in the original paper
        '''
        hyperparameters = self.hyperparameters()
        model = RandomForestClassifier(max_depth= hyperparameters.trees, n_estimators = hyperparameters.depth)
        model.fit(X, y)
        return model

    def test (self, model, X):
        '''
        run inference on a trained model to obtain window states predictions
        inputs:
        model: trained classifier
        X: variable with features from the test set
        '''
        preds = model.predict(X)
        return preds
