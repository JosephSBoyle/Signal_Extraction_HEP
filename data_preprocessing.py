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

from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta

from xgboost import XGBClassifier
from sklearn.model_selection import KFold

import seaborn as sns
import json

from utilities import *
import sys


def re_label(current_label, test_label, train_label):
    """
    :param current_label: event label e.g "ttbar"
    :return: corresponding integer value for that label
    """
    # Signal Labels ('1000', '1400', '300', '400', '440', '460', '500',
    #        '600', '700', '800', '900')
    if current_label in train_label:
        return 1
    elif current_label in test_label:
        return -1
    elif current_label in ("Zlljet", "zlljet"):
        return 0
    elif current_label == "ttbar":
        return 0
    elif current_label == "Diboson":
        return 3
    elif current_label == "singletop":
        return 4


def prepare_data(df, drop_labels, test_label, train_label, drop_neg=True, debug=False, edit_columns=True,
                 relabelling_func=re_label):
    """
    1) Replaces the index with a column 'label'
    2) Converts 'label' into integers by calling re_label
    :param relabelling_func:
    :type debug: object
    :param df: dataframe to preprocess
    :param drop_labels: labels to remove, e.g diboson events
    :param drop_neg: If True, drop negatively weighted samples
    :param debug: if True, samples small amount of data
    :param edit_columns: if True, change format of column names to remove "b" from beginning of string.
    & weight -> EventWeight
    :return: modified dataframe.
    """
    if 'EventWeight' in df.columns:
        edit_columns = False

    if edit_columns:
        for col in df.columns:
            df.rename(columns={col: str(col)[2:-1]}, inplace=True)
            df.rename(columns={'weight': 'EventWeight'}, inplace=True)

    df = df.sample(frac=np.float([0.01 if debug is True else 1][0]))  # shuffle

    ind = pd.Series(df.index)
    df = df.reset_index()

    df['label'] = ind.map(lambda x: relabelling_func(x, test_label, train_label))  # text -> numeric labels

    for i in drop_labels:
        df = df[df.label != i]

    df = df.dropna(axis=0)  # drop columns with NaNs

    if drop_neg:
        df = df[df['EventWeight'] > 0]
    df = df.reset_index(drop=True)
    return df


def rescale(df, variables, target_var='label', transform=False):
    """
    :param transform: scale data to 0-1 range
    :param df: data.
    :param variables: MUST contain EventWeight as last variable.
    :param target_var: sample labels.
    :return:
    """

    y = df[target_var]

    X = df[variables]
    X = np.array(X)

    X_weights = X[:, -1]
    assert min(X_weights) >= 0
    X = X[:, :-1]

    if transform:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    return np.array(X), np.array(y), np.array(X_weights)


def background_split(background_df, weight_col='EventWeight'):
    tr = background_df.sample(frac=0.5)
    te = background_df.drop(tr.index)
    tr[weight_col] *= 2
    tr[weight_col] *= 2
    # independent background events for train and test datasets.
    return tr, te
