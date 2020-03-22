import warnings
from decimal import *
from typing import Dict, Union

import seaborn as sn

import pandas as pd
import numpy as np
from numpy import matrix, array, shape

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

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
import sys


def train_model(Xtrain, ytrain, Xtr_weights):
    xgb = XGBClassifier(subsample=0.5)  # tree_method="hist", grow_policy="lossguide", verbose_eval=1, max_depth=5,
    # x = len(ytrain)
    #x = 1000
    print(f"Training BDT")
    start = timer()
    xgb.fit(Xtrain, ytrain, sample_weight=Xtr_weights)
    end = timer()
    print(timedelta(seconds=end - start))
    return xgb


def test_model(fit_model, X, y, w, frac=None, debug=False):
    if debug:
        X = X[:1000]
        y = y[:1000]

    if "tensorflow" in str(type(fit_model)):
        print("tf model")
        ypred = fit_model.predict(X).argmax(axis=1)
    else:
        print("xgb model")
        ypred = fit_model.predict(X)
        ypred = ypred.round(0)

    # sum of weight in the Signal-like data for y: [B,S]
    signal_sum = 0
    backg_sum = 0

    for c, (i, j) in enumerate(zip(ypred, y)):
        if i == 1:
            if j == 1:
                signal_sum += w[c]
            else:
                backg_sum += w[c]

    print(signal_sum)
    if frac:
        signal_sum *= 1 / frac

    print(signal_sum)
    # backg_sum *= 1/(s_samples/b_samples)
    try:
        purity = signal_sum / (backg_sum+signal_sum)
        significance = signal_sum / np.sqrt(backg_sum+signal_sum)
    except ZeroDivisionError:
        purity = 0
        significance = 0
    results: Dict[str, Union[float, int]] = {'FoM: S / sqrt(S + B)': significance, 'FoM: S / (S + B)': purity
                                             ,'Signal weight "TP"': signal_sum,
               'Background weight "FP"': backg_sum}

    print(f"SIGNAL SIGNIFICANCE: {significance}")
    print(f"tp weight: {signal_sum},  fn weight: {backg_sum}")
    return results


def threshold_weighting(fit_model, X, y, w, threshold, boost=None):
    ypred = fit_model.predict_proba(X)
    ypred = (ypred[:, 1] >= threshold).astype(bool)
    signal_w, backg_w = 0, 0

    for c, (i, j) in enumerate(zip(ypred, y)):
        if i == 1:
            backg_w += w[c]
        else:
            signal_w += w[c]

    if boost is None:
        return signal_w, backg_w
    else:
        return signal_w / boost, backg_w
