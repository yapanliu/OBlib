from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio
import tensorflow as tf
import logging


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class Template():
    '''
    put some model description here
    '''

    class hyperparameters:
        '''
        optimal hyperparameters as presented in the paper
        '''


    def preprocess_data():
        '''
        use this function to do all preprocessing related to the used method
        '''
        data = 0
        return data

    def train(self, X, y):
        '''
        train the model
        '''
        return model

    def domain_adaptation():
        '''
        run domain adaptation for
        '''
        data = 0
        return data

    def test(self, model, X):
        '''
        test the model
        '''
        # model = self.train(X, y)
        preds = model.predict(X)
        return preds




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

    class FeatureNames:
        '''
        this is examplarly done for the case of ashrae OB database study #26
        remarks:
        'BLINDSState_rightNow' is removed, since only 30 points are recorded
        '''
        feature_strings = ['Indoor_Temp [C]', 'Outdoor_Temp [C]',
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


    def preprocess_data():
        '''
        use this function to do all preprocessing related to the used method
        '''
        data = 0
        return data

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

