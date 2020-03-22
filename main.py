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
from threshold_analysis import threshold_metrics
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
train_labels = ('300', '400', '440', '460', '500', '600', '700', '800', '900', '1000', '1400')
test_label = '500'
data = prepare_data(pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\Updated_data.pkl"),
                    drop_labels=config['drop_labels'], train_label=train_labels, test_label=test_label)

train_df = data[data['label'] != -1]

test_sg = data[data['label'] == -1]
test_sg['label'] = test_sg['label'].map(lambda x: 1 if x == -1 else 0)
test_bg = train_df[train_df['label'] == 0].sample(frac=0.1)
test_df = pd.concat([test_sg, test_bg], ignore_index=True)

print(test_df['label'].unique())
print(len(train_df), len(test_df))

print(data.columns, len(data.columns))


# test range of 200-1000GeV
limits = np.arange(0.2e6, 1e6, 0.05e6)
print(limits)

fig = plt.figure(figsize=(20, 20))
gs = gridspec.GridSpec(2, 1, height_ratios=[5, 3])  # vertically stacked plots
ax1 = plt.subplot(gs[0])
ax3 = plt.subplot(gs[1], sharex=ax1)

mVHres_importance = []
FoM_score = []
bin_mid = []

for x in range(len(limits) - 1):
    bin_mid.append((limits[x] + limits[x + 1]) / 2)
    print(limits[x], limits[x + 1])
    temp_df, frac = balance_and_flatten(data, limits=(limits[x], limits[x + 1]))

    X, y, w = rescale(temp_df, config['variables'])
    Xtrain, Xtest, ytrain, ytest, Xtr_weights, Xte_weights = train_test_split(X, y, w, test_size=0.2, random_state=42)

    mva = train_model(Xtrain, ytrain, Xtr_weights)
    mVHres_importance.append(mva.feature_importances_[3])
    print(mva)
    results = (test_model(mva, Xtest, ytest, Xte_weights, frac=frac, debug=False))
    FoM_score.append(results['FoM: S / (S + B)'])
    print(config["variables"], config["variables"][3])
    print("mVHres feature importance: ", mva.feature_importances_[3], "\n\n\n")

ax3.grid(True)
ax3.scatter(bin_mid, FoM_score)
ax3.set_ylabel("Signal Purity:\n S / (S + B)", fontsize=26)
ax3.set_xlabel("Mass / GeV", fontsize=26)

ax2 = ax1.twinx()

ax2.set_title("Binned Models", fontsize=30)
ax1.set_ylabel("Sample weight", fontsize=26)

ax2.set_xlabel("Mass / GeV", fontsize=26)
ax2.tick_params(axis="x", labelsize=20)

ax2.set_ylabel("mVHres importance", color='tab:blue', fontsize=30)  # we already handled the x-label with ax1
ax2.scatter(x=bin_mid, y=mVHres_importance, s=[550 for i in range(len(bin_mid))], marker="x", color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

ticks = ax1.get_xticks()*10**-3

ax1.set_xticklabels(ticks)
ax2.set_xticklabels(ticks)
ax3.set_xticklabels(ticks)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

figure = plt.gcf()  # get current figure
figure.set_size_inches(30, 10)
# legend_without_duplicate_labels(ax1)

plt.savefig(f"{os.getcwd()}.\\binned_models.png", dpi=1200, format='png')
plt.show()

###

X, y, w = rescale(train_df, config['variables'])
mva = train_model(X, y, w)

X_te, y_te, w_te = rescale(train_df, variables=config['variables'], transform=False)
plt.figure(figsize=(20, 10))
cutoffs = np.arange(0, 1, 0.05)
signal = []
back = []
all_ = []
for x in cutoffs:
    midpoints, sg_bar, bg_bar, Sig, Pur, Eff = threshold_metrics(mva, X_te, y_te, w_te)
    plt.figure(figsize=(20, 10))
    plt.title("Threshold Analysis 500GeV test data")

    gs = gridspec.GridSpec(3, 1, height_ratios=[8, 4, 4])  # vertically stacked plots

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax4 = ax3.twinx()

    ax1.set_yscale('log')
    b = ax1.plot(midpoints, bg_bar, ls="steps", color="black")
    s = ax1.plot(midpoints, sg_bar, ls="steps", color="g")
    ax1.legend((s[0], b[0]), ("Signal", "Background"))

    ax1.set_ylabel("Net event Weight")

    ax2.plot(midpoints, Sig)
    ax2.set_ylabel("Signal Significance")

    ax3.plot(midpoints, Pur, color="purple")
    ax3.set_ylabel("Signal Purity")
    ax3.set_xlabel("Output threshold")

    eff_colour = "fuchsia"
    ax4.plot(midpoints, Eff, color=eff_colour)
    ax4.tick_params(axis='y', labelcolor=eff_colour)
    ax4.set_ylabel("Signal Efficiency", color=eff_colour)

    ymin, ymax = ax1.get_ylim()
    ax1.axvline(midpoints[Sig.index(max(Sig))], ymin=0, ymax=ymax, color='r', ls='-')
    ax2.axvline(midpoints[Sig.index(max(Sig))], ymin=0, ymax=1, color='r', ls='-')
    ax3.axvline(midpoints[Sig.index(max(Sig))], ymin=0, ymax=1, color='r', ls='-')

    z = np.arange(0, 1.05, 0.05)
    ax1.set_xticks(z)
    ax2.set_xticks(z)
    ax3.set_xticks(z)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    print("Maximum Sig of all thresholds: ", max(Sig))
    plt.show()

    ax = plt.figure(figsize=(18, 14))
    features = config['variables'][:-1]
    print(len(features))
    print(len(xgb.feature_importances_))
    plt.xticks(rotation=90)
    plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
    plt.xticks(np.arange(0, len(features), 1), labels=features)
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.title("Feature Importance: Merged Signal")
    plt.show()