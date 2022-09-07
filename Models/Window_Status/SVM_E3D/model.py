import pandas as pd
import pickle
from sklearn.svm import SVC


class Model:
    """
    model description
    """
    def __init__(self):
        """
        hyperparameters if needed
        """
        self.hyperparameters = {'gamma': 1,
                                'C': 2}

    class FeatureNames:
        """
        naming of the required features for the random forest model in the
        study 26 from the data ob database

        remarks:
        'BLINDSState_rightNow' is removed, since only 30 points are recorded
        """
        feature_strings = ['Date_Time', 'Indoor_Temp [C]', 'Outdoor_Temp [C]',
                           'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']
        target_string = ['Windor_Status']

    def dataset(self):
        dataset_name = 'ASHRAE Study 26'
        dataset_path = 'data/study_26/Study_26_Study26.csv'
        return dataset_name, dataset_path

    def preprocess_data(self, df):
        """
        use this function to do all preprocessing related to the used method
        """
        df = df[
            ['Date_Time', 'Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]',
             'OccupantNumber', 'Windor_Status']]
        df_nonan = df.dropna()

        # ToDo: proper DateTime conversion for later preprocessing (including time)
        df_nonan['Date_Time_pd'] = pd.to_datetime(df_nonan['Date_Time'], dayfirst=True)
        df_nonan = df_nonan.sort_values('Date_Time_pd', ascending=True)
        df_nonan = df_nonan.sort_values(by=['Date_Time_pd'])
        df_nonan = df_nonan.drop(columns=['Date_Time_pd'])

        dates = ['01-Jul', '02-Jul', '03-Jul', '04-Jul', '05-Jul', '06-Jul', '07-Jul',
                 '01-Jan', '02-Jan', '03-Jan', '04-Jan', '05-Jan', '06-Jan', '07-Jan',
                 '01-Apr', '02-Apr', '03-Apr', '04-Apr', '05-Apr', '06-Apr', '07-Apr',
                 '01-Oct', '02-Oct', '03-Oct', '04-Oct', '05-Oct', '06-Oct', '07-Oct']

        for current_date in dates:
            if current_date in ['01-Jul']:
                df_train = df_nonan[df_nonan['Date_Time'].str.contains(current_date)]
                df_test = df_nonan[df_nonan['Date_Time'].str.contains(current_date) == False]

            else:
                df_train_ = df_nonan[df_nonan['Date_Time'].str.contains(current_date)]
                df_test = df_test[df_test['Date_Time'].str.contains(current_date) == False]

                df_train = pd.concat([df_train, df_train_])

        # ToDo: currently switched as not sure if 10:90 split (train to test) by mistake
        x_train = df_test[
            ['Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']]
        y_train = df_test[['Windor_Status']]

        x_test = df_train[
            ['Indoor_Temp [C]', 'Outdoor_Temp [C]', 'Outdoor_Air_Speed [m/s]', 'OUTDoor_RH [%]', 'OccupantNumber']]
        y_test = df_train[['Windor_Status']]
        test_time = df_train[['Date_Time']]
        return x_train, y_train, x_test, y_test, test_time

    def train(self, df, path):
        """
        to showcase training procedure or for local training
        please add used method for saving trained model
        """
        x_train, y_train, x_test, y_test, test_time = self.preprocess_data(df)

        model = SVC(kernel='rbf', gamma=self.hyperparameters['gamma'], C=self.hyperparameters['C'])
        model.fit(x_train, y_train)

        if not self.pretrained:
            pkl_path = path / 'model.pkl'
            with open(pkl_path, 'wb') as file:
                pickle.dump(model, file)
        return model

    def load_trained(self, path):
        """
        depending on models type add loading function
        """
        pkl_path = path / 'model.pkl'
        with open(pkl_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def test(self, df, model):
        """
        run inference on a trained model to obtain predictions
        inputs:
        model:
        x: variable with features from test set
        """
        x_train, y_train, x_test, y_test, test_time = self.preprocess_data(df)

        y_pred = model.predict(x_test)
        return y_pred, y_test, test_time
