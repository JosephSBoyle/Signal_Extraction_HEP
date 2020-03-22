from utilities import *
from model import *
from data_preprocessing import rescale
import sys

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


def flatten(signal_df, var, weight, partitions, manual_lims=None, background_df=None):
    """
    function for flattening the distribution of sample weights over a range such that the integral of weight over
    each bin is equal & the integral remains the same.
    :param signal_df: dataframe of signal events you wish to flatten
    :param background_df: dataframe of background_df events to cut
    :param var:
    :param weight:
    :param partitions:
    :param manual_lims:
    :param background_df: background_df df to cut at manual lims if background_df not None.
    :return: signal_df, background_df if background_df not None, else signal_df.
    """
    if manual_lims is None:
        upper_bound, lower_bound = signal_df[var].max(), signal_df[var].min()
    else:
        upper_bound, lower_bound = manual_lims[1], manual_lims[0]

    # we use the weight of all signal events, to keep our signal/background_df weight ratio constant.
    tot_weight = signal_df.sum()[weight]

    # remove events outside this region
    signal_df = signal_df[(signal_df[var] > lower_bound) & (signal_df[var] < upper_bound)]

    # interval per bin
    dx = (upper_bound - lower_bound) / partitions
    # desired weight per bin
    targ_bin_weight = tot_weight / partitions

    print("targ bin w: ", targ_bin_weight)

    # partitions + 1 since there are n+1 limits for n bins.
    bin_lims = [lower_bound + i * dx for i in range(0, partitions + 1)]
    print("bin limits: ", bin_lims)
    dist = []

    # iterate through each bin, sequentially
    for i in range(0, partitions):
        # upper and lower limits for the i'th bin
        ll = bin_lims[i]
        ul = bin_lims[i + 1]

        bin_w = signal_df[(signal_df[var] > ll) & (signal_df[var] < ul)].sum()[weight]

        frac = np.float(Decimal(targ_bin_weight) / Decimal(bin_w))

        z = signal_df

        mask = ((signal_df[var] > ll) & (signal_df[var] < ul))
        z_valid = z[mask]
        z.loc[mask, weight] = z_valid[weight] * frac
        signal_df = z

        bin_w = signal_df[(signal_df[var] > ll) & (signal_df[var] < ul)].sum()[weight]

        dist.append(bin_w)
        try:
            assert bin_w != 0.00
        except assertionError:
            warnings.warn("NOT PROPERLY NORMALISED; EMPTY BIN DETECTED")

    if background_df is not None:
        background_df = background_df[(background_df[var] > lower_bound) & (background_df[var] < upper_bound)]
        return signal_df, background_df
    return signal_df


def balance_and_flatten(df, limits, partitions=10):
    signal_ = df[df.label == 1]
    background_ = df[df.label == 0]

    flat_signal, cut_background = flatten(signal_, 'mVHres', 'EventWeight', partitions=partitions,
                                          background_df=background_, manual_lims=limits)

    signal_total_weight = sum(flat_signal['EventWeight'])
    background_total_weight = sum(cut_background['EventWeight'])

    frac = background_total_weight / signal_total_weight

    print(len(cut_background) / len(flat_signal))
    print("boosted signal weight: ", signal_total_weight * frac)
    print("background weight : ", background_total_weight)

    flat_signal['EventWeight'] = flat_signal['EventWeight'] * frac

    df = pd.concat([flat_signal, cut_background])

    # shuffle 5 times
    for i in range(5):
        df = df.sample(frac=1)
    print(f"\n\nS & B balanced in range {limits}")
    return df, frac


def match_background(train_df, limits, variables, partitions=20, to_df=False):
    X_tr, y_tr, w_tr = None, None, None
    df = None
    for x in range(len(limits)-1):
        ll = limits[x]
        ul = limits[x+1]
        print(ll, ul)
        temp_df, _ = balance_and_flatten(train_df, limits=(ll, ul), partitions=partitions)
        X, y, w = rescale(temp_df, variables, transform=False)
        print(len(y))
        if to_df:
            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])
        else:
            if X_tr is not None:
                print("JOINING ARRAYS")
                X_tr = np.concatenate([X_tr, X])
                y_tr = np.concatenate([y_tr, y])
                w_tr = np.concatenate([w_tr, w])
            else:
                print("INITIATING ARRAYS")
                X_tr = X
                y_tr = y
                w_tr = w

    if to_df:
        return df
    else:
        return X_tr, y_tr, w_tr


def plot_weight(plot_df, prev_df, var, weight, flat_range, axes, partitions=10, manual_lims=None):
    if manual_lims is None:
        upper_bound, lower_bound = max(prev_df[var].max(), plot_df[var].max()), min(prev_df[var].min(),
                                                                                    plot_df[var].min())
    else:
        upper_bound, lower_bound = manual_lims[1], manual_lims[0]

    # interval per bin
    dx = (upper_bound - lower_bound) / partitions
    width_ = dx
    # partitions + 1 since there are n+1 limits for n bins.
    bin_lims = [lower_bound + i * dx for i in range(0, partitions + 1)]

    dist = []
    old_dist = []

    # iterate through each bin, sequentially
    for i in range(0, partitions):
        # upper and lower limits for the i'th bin
        ll = bin_lims[i]
        ul = bin_lims[i + 1]

        bin_w = plot_df[(plot_df[var] > ll) & (plot_df[var] < ul)].sum()[weight]
        old_df_bin_w = prev_df[(prev_df[var] > ll) & (prev_df[var] < ul)].sum()[weight]

        print("bin ", i + 1, bin_w)
        dist.append(bin_w)
        old_dist.append(old_df_bin_w)

    bins_midpoints = [bin_lims[i] + dx / 2 for i in range(0, len(bin_lims) - 1)]
    print(len(bin_lims))

    axes.bar(bins_midpoints, old_dist, width=width_, color='r', label="Background")

    axes.bar(bins_midpoints, dist, width=width_,
             label=f"Signal: {int(flat_range[0]/1e3)}-{int(flat_range[1]/1e3)} / GeV", alpha=0.5, hatch="/")

    print(sum(dist), sum(old_dist))
    return

