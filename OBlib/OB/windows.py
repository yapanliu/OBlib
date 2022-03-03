from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class LogisticRegression_Haldi():
    '''
    window opening model proposed by Haldi and Robinson (2009)
    https://www.sciencedirect.com/science/article/pii/S0360132309000961?casa_token=ndqu2NshapsAAAAA:HV-Gcjpdb_wnOvYT8XJjrhetbS2q15phzCrzF41us0iPdQzpnoMiUcBUuoBN1WzwmNhbKPnuets
    '''
    class parameters:
        '''
        parameters as fitted using the original model
        '''
        beta1 = -0.1814
        beta2 = 0.14477
        intercept = 1.459

    def predict(self,X):
        '''
        infer the window states using logistic regression as presented in the
        original paper
        X: indoor air temperature, outdoor air temperature
        '''
        model = LogisticRegression()
        parameters = self.parameters()
        model.coef_ = np.array([[parameters.beta1, parameters.beta2]])
        model.intercept_ = parameters.intercept
        model.classes_ = np.array([0, 1])
        print('coefficients: ', model.coef_[0][0])
        print('bias: ', model.intercept_)
        preds = model.predict(X)
        return preds

class LogisticRegression_default():
    '''
    window opening model proposed by Haldi and Robinson (2009)
    https://www.sciencedirect.com/science/article/pii/S0360132309000961?casa_token=ndqu2NshapsAAAAA:HV-Gcjpdb_wnOvYT8XJjrhetbS2q15phzCrzF41us0iPdQzpnoMiUcBUuoBN1WzwmNhbKPnuets
    '''
    class parameters:
        '''
        parameters as fitted using the original model
        '''
        beta1 = -0.1814
        beta2 = 0.14477
        intercept = 1.459

    def predict(self,X, y):
        '''
        infer the window states using logistic regression as presented in the
        original paper
        X: indoor air temperature, outdoor air temperature
        '''
        model = LogisticRegression()
        model.fit(X.iloc[100000:110000, :], y.iloc[100000:110000])
        parameters = self.parameters()

        model.classes_ = np.array([0, 1])
        print('coefficients: ', model.coef_[0][0])
        print('bias: ', model.intercept_)
        preds = model.predict(X)
        return preds


class RandomForest_E3D():
    '''
    random forest for window states classification
    '''
    class hyperparameters:
        '''
        optimal hyperparameters as presented in the paper
        '''


    def train(self, X, y):
        '''

        '''
        model = RandomForestClassifier(max_depth=2, n_estimators = 100)
        model.fit(X, y)
        preds = model.predict(X)
        return model

    def predict (self, model, X):
        '''
        
        '''
        #model = self.train(X, y)
        preds = model.predict(X)
        return preds



