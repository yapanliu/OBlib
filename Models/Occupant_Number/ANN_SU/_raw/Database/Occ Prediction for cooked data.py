#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[12]:


trainging_X = pd.read_pickle(r"C:\Users\J_zer\OneDrive\Desktop\Database\trainging_X.pickle")
trainging_y = pd.read_pickle(r"C:\Users\J_zer\OneDrive\Desktop\Database\trainging_y.pickle")
testing_X = pd.read_pickle(r"C:\Users\J_zer\OneDrive\Desktop\Database\testing_X.pickle")
testing_y = pd.read_pickle(r"C:\Users\J_zer\OneDrive\Desktop\Database\testing_y.pickle")


# In[13]:


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


# In[14]:


path = r"C:\Users\J_zer\OneDrive\Desktop\Database\Occ_Prediction"
model.load_state_dict(torch.load(path))
model.eval()


# In[15]:


with torch.no_grad():

    y_15 = model(torch.from_numpy(testing_X).float())
    y_15 = y_15.to('cpu')
    y_15 = np.around(y_15).detach().cpu().numpy()


# In[16]:


fig, ax = plt.subplots(figsize=(28,9),dpi=250)
ax.plot_date(testing_y.index, testing_y, '-' , color='lightcoral',linewidth=4,drawstyle='steps-post')
ax.plot_date(testing_y.index, y_15, '-' , color='#0396A6',linewidth=4,drawstyle='steps-post')

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

plt.show()


# In[ ]:




