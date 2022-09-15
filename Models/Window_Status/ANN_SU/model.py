# -*- coding: utf-8 -*-

# Loading all Libs
from opcode import cmp_op
from turtle import color
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import pickle
import io
import scipy.stats as stats
from pathlib import Path 
import sklearn.metrics
from evaluation import AbsoluteMetrices

# use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Winter Split Data contains the data from November, December, January, February and, March
# Transition season Data contains the data from April, May, September, October
# August is not taken into consideration because the apartment are empty for most of the time.

df = pd.read_csv('./Models/Window_Status/ANN_SU/_raw/MASTER_DATASET_FOR_Philadelphia_data.csv')
df = df.dropna()

# Applying Z score to all the predictors in the data frame to normalize the data
df.iloc[:,1:6] = df.iloc[:,1:6].apply(stats.zscore)
X = df.drop(['Window_Status'], axis=1).values
y = df['Window_Status'].values

# Splitting the data randomly into traning and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# create ANN model
def createANNmodel(learningRate):

  # model architecture
  
  ANNclassify = nn.Sequential(
      nn.Linear(6,60),          # input layer
      nn.ReLU(),                # activation unit
      nn.Linear(60,40),         # input layer
      nn.ReLU(),                # activation unit
      nn.Linear(40,26),         # input layer
      nn.ReLU(),                # activation unit
      nn.Linear(26,17),         # output unit
      nn.ReLU(),                # activation unit
      nn.Linear(17,1)           # output unit
      )

  # loss function
  lossfun = nn.BCEWithLogitsLoss()

  # optimizer
  optimizer = torch.optim.Adam(ANNclassify.parameters(),lr=learningRate)
  
  ANNclassify.to(device)
  # model output
  return ANNclassify,lossfun,optimizer

# load the saved model 
def load_model(path):
    # load saved torch model
    # open pickle file
    with open(path, 'rb') as f:
        ANNmodel = pickle.load(f)
    
    ANNmodel.to(device)
    
    # final forward pass
    predictions = ANNmodel(X_train)

    totalaccTraining = 100*torch.mean(((predictions>0) == y_train.unsqueeze(1).float()).float())

    predictions_for_testing = ANNmodel(X_test)

    predictions_for_Initial_Table_again = ANNmodel(X)

    totalaccTesting = 100*torch.mean(((predictions_for_testing>0) == y_test.unsqueeze(1).float()).float())

    return predictions,predictions_for_testing, predictions_for_Initial_Table_again, totalaccTraining, totalaccTesting

# data to GPU
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

X = X.to(device)
y = y.to(device)

# load model
# model_path = "./Models/Window_Status/DNN_SU/model.pickle"
model_path = "./Models/Window_Status/ANN_SU/model.pkl"
predictions,predictions_for_testing, predictions_for_Initial_Table_again, totalaccTraining, totalaccTesting = load_model(model_path)

# data to cpu
X_train = X_train.to('cpu')
y_train = y_train.to('cpu')

X_test = X_test.to('cpu')
y_test = y_test.to('cpu')

X = X.to(device)
y = y.to(device)

# plt.plot(predictions.cpu().detach())

# plt.plot(predictions.cpu().detach()[1000:2000])

a = pd.DataFrame(X_test.numpy())
b = pd.DataFrame(y_test.numpy())
a["Y_real"] = b
c = pd.DataFrame(predictions_for_testing.cpu().detach().numpy())
a["Predictions_Raw"] = c
a["Predictions_Raw"][a["Predictions_Raw"] >= 0] = 1
a["Predictions_Raw"][a["Predictions_Raw"] < 0] = 0

a1 = pd.DataFrame(X.cpu().numpy())
b1 = pd.DataFrame(y.cpu().numpy())
a1["Y_real"] = b1 
c1 = pd.DataFrame(predictions_for_Initial_Table_again.cpu().detach().numpy())
a1["Predictions_Raw"] = c1
a1["Predictions_Raw"][a1["Predictions_Raw"] >= 0] = 1
a1["Predictions_Raw"][a1["Predictions_Raw"] < 0] = 0

# fig, axs = plt.subplots(2)
# axs[0].plot(a.index.values, a["Y_real"])
# axs[0].set_title("Real Values")
# axs[0].title.set_size(16)
# axs[0].set_title("Real Test Values")

# axs[1].plot(a.index.values, a["Predictions_Raw"])
# axs[1].set_title("Predicted Values for Test Set")
# axs[1].title.set_size(16)

# fig.set_figheight(8)
# fig.set_figwidth(16)

# plt.savefig('./Models/Window_Status/DNN_SU/test_results/fig1.png')

# plt.show()

fig, axs = plt.subplots(2)
axs[0].plot(a1.index.values, a1["Y_real"], color='#636EFA')
axs[0].set_title("Real Values")
axs[0].title.set_size(16)

axs[1].plot(a1.index.values, a1["Predictions_Raw"], color='#EF553B')
axs[1].set_title("Predicted Values")
axs[1].title.set_size(16)

fig.set_figheight(8)
fig.set_figwidth(16)

plt.savefig('./Models/Window_Status/DNN_SU/test_results/fig.png')
# plt.show()

# save the test and prediction values
df_evaluations = a[["Y_real", "Predictions_Raw"]].copy()
df_evaluations.columns = ["Y_real", "Predictions_Raw"]
df_evaluations['Predictions_Raw'] = df_evaluations['Predictions_Raw'].astype(int)
df_evaluations['Y_real'] = df_evaluations['Y_real'].astype(int)
df_evaluations.reset_index(inplace=True, drop=True)

# evaluation metrics
metrices = AbsoluteMetrices(df_evaluations['Y_real'], df_evaluations['Predictions_Raw'])
eval_fetch = getattr(metrices, 'Window_Status')
eval = eval_fetch()

# results_dir = "./Models/Window_Status/DNN_SU/test_results/"

# with open(f'{results_dir}/evaluation.pickle', 'wb') as f:
#     pickle.dump(eval, f)
    
# with open(f'{results_dir}/predictions.pickle', 'wb') as f:
#     pickle.dump(df_evaluations, f)

# check raw data
# df['Date_Time'] = pd.to_datetime(df['Date_Time'])
# df['Date_Time'].sort_values(ascending=True)
# df['Date_Time'].sort_values(ascending=False)
# df['Date_Time'].dt.date.unique() # 2012-07-30 - 2013-07-30