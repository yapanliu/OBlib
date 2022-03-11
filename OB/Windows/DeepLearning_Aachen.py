from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio
import tensorflow as tf
import logging
import pandas as pd

class PrepareData():
    '''

    '''
    class FeatureNames:
        feature_strings = ['Date_Time', 'Indoor_Temp [C]', 'Outdoor_Temp [C]',
                           'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']
        target_string = ['Windor_Status']

    def process_data():
        '''
        data preprocessing for the OB model
        '''


class Model():
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

