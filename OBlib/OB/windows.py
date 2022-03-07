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


class ModelTemplate():
    '''
    model description
    '''

    class hyperparameters:
        '''
        hyperparameters from the original paper
        '''

    class FeatureNames:
        '''
        which features are required for the current model?
        '''


    def preprocess_data(self, ):
        '''
        da
        '''
        return

    def train(self, X, y):
        '''
        train the model
        '''
        return model

    def load_trained_model(self, path):
        '''
        train the model
        '''
        return model

    def domain_adaptation(model, X):
        '''
        run domain adaptation for pretrained model
        '''
        return model

    def test(self, model, X):
        '''
        test the model
        '''
        # model = self.train(X, y)
        predictions = model.predict(X)
        return predictions




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

    def test(self,X):
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
    def train(self, X, y):
        '''
        train the model
        '''
        model = LogisticRegression()
        model.fit(X, y)
        return model


    def test(self,model, X):
        '''
        infer the window states using logistic regression as presented in the
        original paper
        X: indoor air temperature, outdoor air temperature

        model = self.train(X, y)
        parameters = self.parameters()

        model.classes_ = np.array([0, 1])
        print('coefficients: ', model.coef_[0][0])
        print('bias: ', model.intercept_)
        '''



        preds = model.predict(X)
        return preds


class RandomForest_E3D():
    '''
    toDo: add model description
    '''

    class hyperparameters:
        '''
        optimal hyperparameters as presented in the paper
        '''
        trees = 100
        depth = 2

    def preprocess_data(self, x):
        '''
        use this function to do all preprocessing related to the used method
        '''
        dates = ['01-Jul', '02-Jul', '03-Jul', '04-Jul', '05-Jul', '06-Jul', '07-Jul',
                 '01-Jan', '02-Jan', '03-Jan', '04-Jan', '05-Jan', '06-Jan', '07-Jan',
                 '01-Apr', '02-Apr', '03-Apr', '04-Apr', '05-Apr', '06-Apr', '07-Apr',
                 '01-Oct', '02-Oct', '03-Oct', '04-Oct', '05-Oct', '06-Oct', '07-Oct']
        data_all = []
        for current_date in dates:
            data = x[x['Date_Time'].str.contains(current_date)]
            # print(data)
            # input('test')
            data_all.append(data)
        df = pd.concat(data_all)

        data_all_test = []
        for current_date in dates:
            data = x[x['Date_Time'].str.contains(current_date) == False]
            # print(data)
            # input('test')
            data_all_test.append(data)
        df_test = pd.concat(data_all_test)

        x_train = df[
            ['Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']]
        y_true = df[['Windor_Status']]

        x_test = df_test[
            ['Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']]
        y_test = df_test[['Windor_Status']]

        return x_train, y_true, x_test, y_test

    class FeatureNames:
        '''
        this is examplarly done for the case of ashrae OB database study #26
        remarks:
        'BLINDSState_rightNow' is removed, since only 30 points are recorded
        '''
        feature_strings = ['Date_Time', 'Indoor_Temp [C]', 'Outdoor_Temp [C]',
                           'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber','Windor_Status']
        target_string = ['Windor_Status']


    def train(self, X, y):
        '''
        '''
        hyperparameters = self.hyperparameters()
        model = RandomForestClassifier(max_depth= hyperparameters.trees, n_estimators = hyperparameters.depth)
        model.fit(X, y)
        return model

    def test (self, model, X):
        '''
        '''
        preds = model.predict(X)
        return preds




class DeepLearn():
    logging.getLogger('tensorflow').disabled = True
    '''
    put some model description here
    '''

    class hyperparameters:
        '''
        optimal hyperparameters as presented in the paper
        '''
        # number of hidden neurons
        hidden_neurons = [64, 94, 81, 10, 25]
        # learning rate
        lr = 0.1
        # regularization
        reg = 1 / 10 ** (4)
        # define devices used for adaptation
        device = '/cpu:0'
        # number of training iterations (in thousands)
        nr_iter = 8



    def load_trained_model(self, path):
        '''
        function used to load the pretrained model. the loaded model will percieve the hyperparameters as defined by the
        original model training. In case of further development or fine tunings, the updated hyperparameters could be defined
        as the inputs to the tf.contrib.learn.DNNClassifier
        :param path: global path where the trained model was saved
        :return: classifier: trained model that can be used for further evaluation or as a baseline model for further development
        '''
        # use default hyperparameters
        hyperparmeters = self.hyperparameters()
        # data type and dimension of input features
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=25, dtype=tf.float32)]
        # load pretrained model
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=hyperparmeters.hidden_neurons,
                                                    activation_fn=tf.nn.relu,
                                                    optimizer=tf.train.ProximalAdagradOptimizer
                                                    (learning_rate=hyperparmeters.lr,
                                                     l1_regularization_strength=hyperparmeters.reg), model_dir=path)
        return classifier

    def domain_adaptation(self, model, X_, y_):
        '''
        run domain adaptation for
        '''
        '''
        :param model: loaded and defined model
        :param training_set: training set used for weight adapation
        :param training_target: corresponding target
        :param device: used device; for example '/gpu:0' or '/cpu:0'
        :param nr_iter: number of adaptation iterations divided by 1000
        :return: model with adapted weights
        '''
        hyperparmeters = self.hyperparameters()
        with tf.device(hyperparmeters.device):
            for i in range(1, set_parameters.nr_iter + 1):
                model.fit(x=X_, y=y_, steps=1000, batch_size=4096)
                predictions = list(model.predict(X_, as_iterable=True))
        return model

    def test(self, X, model = None):
        '''
        test the model
        '''
        # model = self.train(X, y)
        '''
        run evaluation session and save the rpedicted window states
        :param path_global: global path
        :return:
        '''
        # load input data
        path_model ='./DL_model'
        # start a new tensorflow session
        # with tf.Session() as sess:
        # load trained model
        if model == None:
            model = self.load_trained_model(path_model)
        # make predicitons
        predictions = list(model.predict(X, as_iterable=True))

        return predictions

