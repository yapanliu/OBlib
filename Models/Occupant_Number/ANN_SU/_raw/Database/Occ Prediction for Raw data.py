#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import pickle


# In[2]:


# Density = pd.read_csv(r"C:\Users\J_zer\OneDrive\Desktop\Database\Occupant_Number_Measurement_Study32.csv")
Density = pd.read_csv("/home/yliu88/yapan/github/OBlib/Models/Occupant_Number/ANN_SU/_raw/Occupant_Number_Measurement_Study32.csv")


# In[3]:


Density['Time'] = pd.to_datetime(Density['Date_Time'], format="%Y-%m-%d %H:%M")
Density.set_index("Time", inplace=True)
Density["Time"] = Density.index

d = Density["Time"].dt.dayofweek + 1
h = Density["Time"].dt.hour
m = Density["Time"].dt.minute

ti = h + m / 60
tii = d

Density['Time of day_sin'] = np.sin(ti * (2. * np.pi / 24))
Density['Time of day_cos'] = np.cos(ti * (2. * np.pi / 24))
Density["Day of week"] = d


Density = Density.drop(Density[Density["Day of week"] > 5].index)
Density = Density.drop('Time', 1)
Density_Copy = Density


# In[6]:


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

def train(X_train, Y_train, H1, H2, learning_rate, epochs=2000):
    model = ThreeLayerNet(X_train.shape[1], H1, H2, Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    inputs = torch.from_numpy(X_train).float()
    labels = torch.from_numpy(Y_train).float()
    loss_ = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        loss_.append(loss.item())
        if epoch % 50 == 0:
            print('epoch {}: loss = {}'.format(epoch, loss.item()))
    plt.plot(loss_)
    plt.show()
    return model

df = Density_Copy
pred_15min = []
pred_1h = []
pred_6h = []
pred_24h = []
for i in range(len(df) - 96):
    pred_15min.append(df["Occupant_Number_Measurement"][i + 1])
    pred_1h.append(df["Occupant_Number_Measurement"][i + 4])
    pred_6h.append(df["Occupant_Number_Measurement"][i + 36])
    pred_24h.append(df["Occupant_Number_Measurement"][i + 96])

for s in range(96):
    pred_15min.append("none")
    pred_1h.append("none")
    pred_6h.append("none")
    pred_24h.append("none")

df["Occupancy_pre_15min"] = pred_15min
df["Occupancy_pre_1h"] = pred_1h
df["Occupancy_pre_6h"] = pred_6h
df["Occupancy_pre_24h"] = pred_24h

past_15 = []
past_30 = []
past_15.append("none")
past_15.append("none")
past_30.append("none")
past_30.append("none")
for i in range(2, len(df)):
    past_15.append(df["Occupant_Number_Measurement"][i - 1])
    past_30.append(df["Occupant_Number_Measurement"][i - 2])

df["Occupancy_pas_15"] = past_15
df["Occupancy_pas_30"] = past_30

def rolling(rolling_data):

    roll_train = rolling_data[-144 * 10:-144 * 3]
    roll_test_train = rolling_data[-144 * 3-6*6:-6*6]
    roll_test = rolling_data[-144 * 3:]
    return roll_train, roll_test_train, roll_test


# In[7]:


training, roll_test_train, testing = rolling(df[144*25:144*(28+28)])


# In[25]:


X = training.iloc[:, [1, 4, 5, 10, 12]].to_numpy().astype(float)
Y = training.iloc[:, 9:10].to_numpy().astype(float)

XX = roll_test_train.iloc[:, [1, 4, 5, 10, 12]].to_numpy().astype(float)


# In[26]:


mdl = train(X_train=X, Y_train=Y, H1=20, H2=25, learning_rate=0.001)
with torch.no_grad():

    y_15 = mdl(torch.from_numpy(XX).float())
    y_15 = y_15.to('cpu')
    y_15 = np.around(y_15).detach().cpu().numpy()


# In[27]:


fig = plt.figure(facecolor=(1, 1, 1))
fig, ax = plt.subplots(figsize=(28,9),dpi=250)
ax.plot_date(testing.index, testing.Occupant_Number_Measurement, '-' , color='lightcoral',linewidth=4,drawstyle='steps-post')
ax.plot_date(testing.index, y_15, '-' , color='#0396A6',linewidth=4,drawstyle='steps-post')

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
fig.savefig(r"C:\Users\J_zer\OneDrive\Desktop\Database\6 Hours ahead Occupancy Forcasting.PNG")


# In[32]:


path = r"C:\Users\J_zer\OneDrive\Desktop\Database\Occ_Prediction.pkl"
torch.save(mdl.state_dict(), path)


# In[29]:


trainging_X = X
trainging_y = Y
testing_X = XX
testing_y = testing.Occupant_Number_Measurement


# In[30]:


with open(r"C:\Users\J_zer\OneDrive\Desktop\Database\trainging_X.pickle", 'wb') as handle:
    pickle.dump(trainging_X, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(r"C:\Users\J_zer\OneDrive\Desktop\Database\trainging_y.pickle", 'wb') as handle:
    pickle.dump(trainging_y, handle, protocol=pickle.HIGHEST_PROTOCOL)    
with open(r"C:\Users\J_zer\OneDrive\Desktop\Database\testing_X.pickle", 'wb') as handle:
    pickle.dump(testing_X, handle, protocol=pickle.HIGHEST_PROTOCOL)    
with open(r"C:\Users\J_zer\OneDrive\Desktop\Database\testing_y.pickle", 'wb') as handle:
    pickle.dump(testing_y, handle, protocol=pickle.HIGHEST_PROTOCOL)       
with open(r"C:\Users\J_zer\OneDrive\Desktop\Database\Prediction_Results.pickle", 'wb') as handle:
    pickle.dump(y_15, handle, protocol=pickle.HIGHEST_PROTOCOL)      


# In[31]:





# In[ ]:




