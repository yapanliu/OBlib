import numpy as np
import matplotlib as plt
import csv
import pandas as pd


class load_data:
    def open_file(self, current_file):
        '''
        :param current_file:
        :return:
        '''
        return pd.read_csv(current_file ,sep=r'\s*,\s*',)

    def get_stream(self, data, stream):
        stream1 = data[stream]

        return stream1

    def create_x(self, data, feature_strings):
        x = pd.DataFrame(columns=feature_strings)
        for stream in feature_strings:
            stream1 = self.get_stream(data, stream)  # get current stream
            x[stream] = stream1
        return x








