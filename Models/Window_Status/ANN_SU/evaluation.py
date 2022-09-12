import numpy as np
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
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.Metrices = Metrices(self.y_true, self.y_pred)

    def Occupancy_Measurement(self):
        '''

        '''
        # categorical, binary --> present/not present
        acc = sklearn.metrics.accuracy_score(self.y_true, self.y_pred)
        conf_mat = sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)
        f_1 = sklearn.metrics.f1_score(self.y_true, self.y_pred)
        eval = pd.DataFrame({'Evaluation': [acc, conf_mat, f_1]}, index=['Accuracy', 'Confusion Matrix', 'F1-Score'])
        return eval

    def Occupant_Number(self):
        # continuous --> 0...n
        mae = self.Metrices.mae()
        mse = self.Metrices.mse()
        rmse = self.Metrices.rmse()
        eval = pd.DataFrame({'Evaluation': [mae, mse, rmse]}, index=['MAE', 'MSE', 'RMSE'])
        return eval

    def Window_Status(self):
        # categorical, binary --> open/closed
        # toDo: use metrics from the class Metrics
        acc = sklearn.metrics.accuracy_score(self.y_true, self.y_pred)
        conf_mat = sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)
        f_1 = sklearn.metrics.f1_score(self.y_true, self.y_pred)
        eval = pd.DataFrame({'Evaluation': [acc, conf_mat, f_1]}, index=['Accuracy', 'Confusion Matrix', 'F1-Score'])
        return eval

    def Plug_Load(self):
        # categorical, binary --> in use/not in use
        acc = self.Metrices.acc()
        balanced_acc = self.Metrices.balanced_acc()
        conf_mat = self.Metrices.conf_mat()
        f_1 = self.Metrices.f_1()
        eval = pd.DataFrame({'Evaluation': [acc, balanced_acc, conf_mat, f_1]}, index=['Accuracy', 'Balanced Accuracy', 'Confusion Matrix', 'F1-Score'])
        return eval

    def Shading_Status(self):
        # categorical --> down, up, partially
        acc = self.Metrices.acc()
        conf_mat = self.Metrices.conf_mat()
        f_beta = self.Metrices.f_beta()
        eval = pd.DataFrame({'Evaluation': [acc, conf_mat, f_beta]}, index=['Accuracy', 'Confusion Matrix', 'F-beta Score'])
        return eval

    def Lighting_Status(self):
        # categorical, binary --> on/off
        acc = self.Metrices.acc()
        conf_mat = self.Metrices.conf_mat()
        f_1 = self.Metrices.f_1()
        eval = pd.DataFrame({'Evaluation': [acc, conf_mat, f_1]}, index=['Accuracy', 'Confusion Matrix', 'F1-Score'])
        return eval

    def Thermostat_Adjustment(self):
        # continuous --> settings range [n..m]
        mae = self.Metrices.mae()
        mse = self.Metrices.mse()
        rmse = self.Metrices.rmse()
        n_rmse = self.Metrices.n_rmse()
        eval = pd.DataFrame({'Evaluation': [mae, mse, rmse, n_rmse]}, index=['MAE', 'MSE', 'RMSE', 'N-RMSE'])
        return eval