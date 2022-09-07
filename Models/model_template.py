from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Model:
    """
    model description
    """
    def __init__(self):
        """
        hyperparameters if needed
        """
        self.hyperparameters = {}

    class FeatureNames:
        """
        naming of the required features
        """
        feature_strings = ['', '', '']
        target_string = ['']

    def dataset(self):
        """
        naming of the required dataset (ASHRAE Study XY) and path to said data
        """
        dataset_name = ''
        dataset_path = ''
        return dataset_name, dataset_path

    def preprocess_data(self, df):
        """
        use this function to do all preprocessing related to the used method
        """
        x = []
        y = []
        time = []
        return x, y, time

    def train(self, df):
        """
        to showcase training procedure or for local training
        please add used method for saving trained model
        """
        model = []
        return model

    def load_trained(self, path):
        """
        depending on model type add loading function
        """
        model = []
        return model

    def test(self, df, model):
        """
        run inference on a trained model to obtain predictions
        inputs:
        model:
        x: variable with features from test set
        """
        x_test, y_test, time = self.preprocess_data(df)

        y_pred = model.predict(x_test)
        return y_pred, y_test, time
