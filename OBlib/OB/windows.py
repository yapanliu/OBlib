from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio


from sklearn.linear_model import LogisticRegression


class LR():
    def test_model(self,X,y):
        '''
        window opening model proposed by haldi and robinson (2009)
        '''

        model = LogisticRegression()  # max_iter = 100000,  {0: 10, 1:1} class_weight =  {0: 10, 1:1}
        model.fit(X, y)
        print('validation set defined')
        model.coef_[0][0] = -0.1814
        model.coef_[0][1] = 0.14477
        model.intercept_ = 1.459
        print('coefficients: ', model.coef_[0][0])
        print('bias: ', model.intercept_)
        preds = model.predict(X)

        return preds


