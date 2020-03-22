# author "Joseph S. Boyle"
import warnings
from decimal import *
import seaborn as sn

import pandas as pd
import numpy as np
from numpy import matrix, array, shape

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

from timeit import default_timer as timer
from datetime import timedelta

from xgboost import XGBClassifier
from sklearn.model_selection import KFold

import seaborn as sns
import json

from utilities import *
from data_preprocessing import *
from model import *
from flattening import *
from utilities import legend_without_duplicate_labels
import sys

os.chdir(sys.path[0])

params = {'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

with open("config.json") as json_file:
    config = json.load(json_file)

fig, ax = plt.subplots()

a = ['300', '400', '440', '460', '600', '700']

FoM_score = []
test_mass = []

for i in a:
    print(i)
    print([x for x in a if i != x])
    print("\n\n")

    data = prepare_data(pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\Updated_data.pkl"),
                        drop_labels=config['drop_labels'], test_label=i, train_label=[x for x in a if i != x])

    train_df = data[data['label'] != -1]

    test_sg = data[data['label'] == -1]
    test_sg['label'] = test_sg['label'].map(lambda x: 1 if x == -1 else 0)
    test_bg = train_df[train_df['label'] == 0].sample(frac=0.1)
    train_df = train_df.drop(test_bg.index) # remove test BG events from train_df

    test_df = pd.concat([test_sg, test_bg], ignore_index=True)  # merge training bg and sg events.

    X_te, y_te, w_te = rescale(test_df, config['variables'], transform=False)
    print(np.shape(y_te))
    X_tr, y_tr, w_tr = None, None, None  # not defined yet

    print(test_df['label'].unique())
    print(len(train_df), len(test_df))
    print(data.columns, len(data.columns))

    limits = np.arange(0.3e6, 0.35e6, 0.025e6)

    for x in range(len(limits)-1):
        ll = limits[x]
        ul = limits[x+1]
        print(ll, ul)
        temp_df, _ = balance_and_flatten(train_df, limits=(limits[x], limits[x + 1]), partitions=20)
        X, y, w = rescale(temp_df, config['variables'], transform=False)

        if X_tr is not None:
            print(len(X_tr))
            np.concatenate([X_tr, X])
            np.concatenate([y_tr, y])
            np.concatenate([w_tr, w])
            print(len(X_tr))
        else:
            X_tr = X
            y_tr = y
            w_tr = w

    mva = train_model(X_tr, y_tr, w_tr)
    X_te, y_te, w_te = rescale(test_df, variables=config['variables'], transform=False)
    results = test_model(mva, X_te, y_te, w_te)
    FoM_score.append(results['FoM: S / (S + B)'])
    test_mass.append(i)

ax.plot(test_mass, FoM_score)
plt.show()
