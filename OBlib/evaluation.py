import numpy as np
import matplotlib as plt
import csv
import pandas as pd
import sklearn.metrics


class Metrices:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def acc(self):
        return sklearn.metrics.accuracy_score(self.y_true, self.y_pred)

    def balanced_acc(self):
        return sklearn.metrics.balanced_accuracy_score(self.y_truem, self.y_pred)

    def conf_mat(self):
        return sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)

    def f_1(self):
        return sklearn.metrics.f1_score(self.y_true, self.y_pred)

    def f_beta(self):
        return sklearn.metrics.fbeta_score(self.y_true, self.y_pred)

    def mae(self):
        return np.absolute(np.subtract(self.y_true, self.y_pred)).mean()

    def mse(self):
        return ((np.subtract(self.y_true, self.y_pred)) ** 2).mean()

    def rmse(self):
        '''
        for .mean(...):
        with ax=0 the average is performed along the row, for each column, returning an array
        with ax=1 the average is performed along the column, for each row, returning an array
        with ax=None the average is performed element-wise along the array, returning a scalar value
        '''
        return np.sqrt((np.subtract(self.y_true, self.y_pred) ** 2).mean())

    def n_rmse(self):
        # for normalization rmse can be divided by: mean, max-min, standard deviation, interquartile range q1-q3
        return self.rmse() / self.y_true.mean()


class AbsoluteMetrices:
    '''
    absolute metrices used to evaluate each target OB variable
    '''



    def occupancy(self, y_true, y_pred):
        '''
        
        '''
        # categorical, binary --> present/not present
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
        f_1 = sklearn.metrics.f1_score(y_true, y_pred)
        return acc, conf_mat, f_1

    def occupant_count(self, y_true, y_pred):
        # continuous --> 0...n
        mae = Metrices.mae()
        mse = Metrices.mse()
        rmse = Metrices.rmse()
        return mae, mse, rmse

    def windows(self, y_true, y_pred):
        # categorical, binary --> open/closed
        # toDo: use metrics from the class Metrics
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
        f_1 = sklearn.metrics.f1_score(y_true, y_pred)
        return acc, conf_mat, f_1

    def appliances(self, y_true, y_pred):
        # categorical, binary --> in use/not in use
        acc = Metrices.acc()
        balanced_acc = Metrices.balanced_acc()
        conf_mat = Metrices.conf_mat()
        f_1 = Metrices.f_1()
        return acc, balanced_acc, conf_mat, f_1

    def shadings(self, y_true, y_pred):
        # categorical --> down, up, partially
        acc = Metrices.acc()
        conf_mat = Metrices.conf_mat()
        f_beta = Metrices.f_beta()
        return acc, conf_mat, f_beta

    def lighting(self, y_true, y_pred):
        # categorical, binary --> on/off
        acc = Metrices.acc()
        conf_mat = Metrices.conf_mat()
        f_1 = Metrices.f_1()
        return acc, conf_mat, f_1

    def thermostat(self, y_true, y_pred):
        # continuous --> settings range [n..m]
        mae = Metrices.mae()
        mse = Metrices.mse()
        rmse = Metrices.rmse()
        n_rmse = Metrices.n_rmse()
        return mae, mse, rmse, n_rmse


class RelativeMetrices:
    '''
    relative metrices used to evaluate each target OB variable
    '''
    def occupancy(self, y_true, y_pred):
        return

    def occupant_count(self, y_true, y_pred):
        return

    def windows(self, y_true, y_pred):
        return

    def appliances(self, y_true, y_pred):
        return

    def shadings(self, y_true, y_pred):
        return

    def lighting(self, y_true, y_pred):
        return

    def thermostat(self, y_true, y_pred):
        return

class IndirectMetrices:
    '''
    indirect metrices used to evaluate each target OB variable
    these include for instance the impact the energy consumption,
    subjective measurements etc.
    '''
    def occupancy(self, y_true, y_pred):
        return

    def occupant_count(self, y_true, y_pred):
        return

    def windows(self, y_true, y_pred):
        return

    def appliances(self, y_true, y_pred):
        return

    def shadings(self, y_true, y_pred):
        return

    def lighting(self, y_true, y_pred):
        return

    def thermostat(self, y_true, y_pred):
        return












