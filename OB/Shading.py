from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io as sio
from google.protobuf import text_format


class Occupancy():
    '''
    This model is developed by Mahdavi A, Mohammadi A, Kabir E, Lambeva L (2008)

    referece journal : Occupants’ operation of lighting and shading systems in officebuildings.
                       Journal of building performance simulation, 1(1), 57-65.
                       https://doi.org/10.1080/19401490801906502

    The model uses a deep recurrent network for detecting the number of occupants in a room

    '''
    def run_model():
