#!/usr/bin/env python
# coding: utf-8

import pickle
import json
import requests
from collections import OrderedDict
from pprint import pprint
from datetime import datetime, timedelta, date
from dateutil import parser, tz
import pandas as pd
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib import ticker
import matplotlib.patches as mpatches
import random
import datetime
import itertools
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import dateutil.relativedelta
import time
from evaluation import AbsoluteMetrices


result_dir = "/home/yliu88/yapan/github/OBlib/Models/Occupant_Number/DNN_SU/_raw/"

trainging_X = pd.read_pickle(result_dir + "trainging_X.pickle")
trainging_y = pd.read_pickle(result_dir + "trainging_y.pickle")
testing_X = pd.read_pickle(result_dir + "testing_X.pickle")
y_true = pd.read_pickle(result_dir + "testing_y.pickle")
y_pred = pd.read_pickle(result_dir + "Prediction_Results.pickle")



class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h1 = self.linear1(x).clamp(min=0)
        h2 = self.linear2(h1).clamp(min=0)
        return self.linear3(h2)

model = ThreeLayerNet(5, 20, 25, 1)    


path = result_dir + "model.pkl"
model.load_state_dict(torch.load(path))
model.eval()


with torch.no_grad():

    y_pred = model(torch.from_numpy(testing_X).float())
    y_pred = y_pred.to('cpu')
    y_pred = np.around(y_pred).detach().cpu().numpy()




# save the data
df_select = y_true.squeeze()
df_select = df_select.to_frame()
df_select.reset_index(inplace=True, drop=True)
df_select.columns = ['y_true']
df_select['y_pred'] = y_pred

# change data type to int
df_select[['y_true', 'y_pred']] = df_select[['y_true', 'y_pred']].astype(int)
df_select.clip(lower=0, inplace=True)


# save to pickle files
with open(result_dir + "predictions.pickle", "wb") as f:
    pickle.dump(df_select, f)

y_pred = df_select['y_pred']

# make plots
fig, ax = plt.subplots(figsize=(28,9),dpi=250)
ax.plot_date(y_true.index, y_true, '-' , color='lightcoral',linewidth=4,drawstyle='steps-post')
ax.plot_date(y_true.index, y_pred, '-' , color='#0396A6',linewidth=4,drawstyle='steps-post')

ax.xaxis.set_minor_locator(dates.HourLocator(interval=2))
ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M\n'))
ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n%a\n%b %d'))

plt.setp(ax.xaxis.get_minorticklabels(), rotation=35)

fig.suptitle("6 Hours ahead Occupancy Forcasting", fontsize=28,y=0.93,fontweight='bold')
ax.set_xlabel("Date", fontsize=22,fontweight='bold')
ax.set_ylabel("Occupancy Level", fontsize=22,labelpad=20,fontweight='bold')
ax.tick_params(axis='both', which='minor', labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=20)

ocp_pre = mpatches.Patch(color='#0396A6', label='Groundtruth Data')
ocp_mea = mpatches.Patch(color='lightcoral', label='Prediction Result')

ax.legend(handles=[ocp_pre,ocp_mea],loc="upper left",fontsize=14)

# plt.savefig(result_dir + "fig.png",bbox_inches='tight')

plt.show()


# evaluation metrics
# evaluation metrics
metrices = AbsoluteMetrices(df_select['y_true'], df_select['y_pred'])
eval_fetch = getattr(metrices, 'Occupant_Number')
eval = eval_fetch()

# save evaluation results
with open(result_dir + "evaluation.pickle", "wb") as f:
    pickle.dump(eval, f)
    

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)

