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
        beta1 =  -0.1814
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

class DeepLearn():
    logging.getLogger('tensorflow').disabled = True

    # path where repository is saved
    path_global = 'path-to-cloned-repo/window_opening'
    # path where adaptation features are saved
    path_adaptation_features = 'path-to-my-adaptation-features/my-adaptation-features.mat'
    path_adaptation_labels = 'path-to-my-adaptation-features/my-adaptation-labels.mat'
    path_test_features = 'path-to-my-adaptation-features/my-test-features.mat'
    path_test_labels = 'path-to-my-adaptation-features/my-test-features.mat'

    class session_parameters:
        '''
        parameters for executing the adaptation tensorflow session
        if parameters not defined, default values from this class will be used
        '''
        # define devices used for adaptation
        device = '/cpu:0'
        # number of training iterations (in thousands)
        nr_iter = 8

    class hyperparamters_aachen:
        # number of hidden neurons
        hidden_neurons = [64, 94, 81, 10, 25]
        # learning rate
        lr = 0.1
        # regularization
        reg = 1 / 10 ** (4)

    def load_trained_model(path):
        '''
        function used to load the pretrained model. the loaded model will percieve the hyperparameters as defined by the
        original model training. In case of further development or fine tunings, the updated hyperparameters could be defined
        as the inputs to the tf.contrib.learn.DNNClassifier
        :param path: global path where the trained model was saved
        :return: classifier: trained model that can be used for further evaluation or as a baseline model for further development
        '''
        # use default hyperparameters
        hyperparmeters = hyperparamters_aachen
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

    def weight_adaptation(model, training_set, training_target):
        '''
        :param model: loaded and defined model
        :param training_set: training set used for weight adapation
        :param training_target: corresponding target
        :param device: used device; for example '/gpu:0' or '/cpu:0'
        :param nr_iter: number of adaptation iterations divided by 1000
        :return: model with adapted weights
        '''
        set_parameters = session_parameters
        with tf.device(set_parameters.device):
            for i in range(1, set_parameters.nr_iter + 1):
                model.fit(x=training_set, y=training_target, steps=1000, batch_size=4096)
                predictions = list(model.predict(training_set, as_iterable=True))
        return model

    def make_predictions(path_global):
        '''
        run evaluation session and save the rpedicted window states
        :param path_global: global path
        :return:
        '''
        # load input data
        path_model = path_global + '/trained_model'
        path_features = path_global + '/inputs/features.mat'
        path_target = path_global + '/inputs/labels.mat'
        (x, y) = inputs.get_input_data(path_features, path_target)
        # start a new tensorflow session
        # with tf.Session() as sess:
        # load trained model
        model = load_trained_model(path_model)
        # make predicitons
        predictions = list(model.predict(x, as_iterable=True))
        # save predicitons to a comma separeted txt file
        np.savetxt(path_global + '/output/predictions.txt', predictions, fmt="%f")

    def run_adaptation(path_global, path_adaptation_features, path_adaptation_labels):
        '''
        load the input features and trained model. make predictions of the window states based on the input data
        and save them to a comma separated txt file
        :return:
        '''
        # load adaptation data set
        path_model = path_global + 'OB/DL_model'
        (x_adapt, y_adapt) = inputs.get_input_data(path_adaptation_features, path_adaptation_labels)
        # start a new tensorflow session
        with tf.Session() as sess:
            # load trained model
            model = load_trained_model(path_model)
            # do weight adaptation
            model = weight_adaptation(model, x_adapt, y_adapt)
            # make predicitons
            predictions = list(model.predict(x_adapt, as_iterable=True))
            # test the performace after adaptation
            # load data
            (x_test, y_test) = inputs.get_input_data(path_test_features, path_test_labels)
            # make predicitons
            predictions_test = list(model.predict(x_test, as_iterable=True))
        # save predicitons to a comma separeted txt file
        np.savetxt(path_global + '/output/predictions.txt', predictions_test, fmt="%f")
        np.savetxt(path_global + '/output/labels.txt', y_test, fmt="%f")



