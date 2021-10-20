import numpy as np
import pandas as pd

class occupancy():
    def ashrae(self, x):
        '''
        sample occupancy model
        '''

        x.loc[x['Indoor_CO2[ppm]']  > 600 , 'prediction'] = 1
        x.loc[x['Indoor_CO2[ppm]']  < 600, 'prediction'] = 0
            

        return x['prediction']


