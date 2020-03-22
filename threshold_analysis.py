# author "Joseph S. Boyle"
from utilities import *
from data_preprocessing import *
from model import *
from test_model import filter_unlabelled_data
from flattening import *
from utilities import legend_without_duplicate_labels
from filter import cut_as_filter
import sys
import warnings
from decimal import *
import seaborn as sn

import pandas as pd
import numpy as np
from numpy import matrix, array, shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib import gridspec

from timeit import default_timer as timer
from datetime import timedelta

from xgboost import XGBClassifier
from sklearn.model_selection import KFold

import seaborn as sns
import json


def threshold_metrics(fit_model, X, y, w, limits=np.linspace(0, 1, 10, endpoint=False)):
    """
    Function which generates metrics by analysing the outputs of an MVA.
    :param fit_model: trained model to use for inference
    :param X: data matrix.
    :param y: target labels corresponding to X
    :param w: sample weights. If unweighted, replace with an array of 1's with len(y)
    :param limits: threshold points to evaluate
    :return: Signal: significance, purity and efficiency
    """
    midpoints = []
    sg_bar = []
    bg_bar = []

    median_discovery_significance = []  # sqrt(sum(2*[(S+B)*ln(1+S/B) - S])
    significance = []  # S / root(S+B)
    purity = []  # S/(S+B)
    efficiency = []  # TP / all S

    ypred = fit_model.predict_proba(X)
    print(np.shape(ypred))
    try:
        assert np.mean(ypred.argmax(axis=1)) == np.mean(fit_model.predict(X))
    except AssertionError:
        print("Default threshold != 0.5???\nAre you using multiclass model?")
        print(ypred.argmax(axis=1), fit_model.predict(X))
        pass
    bg_prob = ypred[:, 0]
    sg_prob = ypred[:, 1]

    # for bin in bins:
    for x in range(len(limits) - 1):
        ll = limits[x]
        ul = limits[x + 1]
        midpoints.append((ll + ul) / 2)
        bg_bin_weight = 0
        sg_bin_weight = 0

        for i, (y_, s_, b_, w_) in enumerate(zip(y, sg_prob, bg_prob, w)):
            # if event is in bin: [lower lim, upper lim)
            if ll <= s_ < ul:

                # if event is actually background
                if y_ == 0:
                    bg_bin_weight += w[i]

                # event is signal
                else:
                    sg_bin_weight += w[i]
            else:
                pass

        sg_bar.append(sg_bin_weight)  # /boost
        bg_bar.append(bg_bin_weight)

    # calculate Signal and Background yields for each cut.
    s_above_thresh = []
    b_above_thresh = []

    for cut, (s_, b_) in enumerate(zip(sg_bar, bg_bar)):
        sig_in_cut = 0
        bac_in_cut = 0
        for subcut, (i, j) in enumerate(zip(sg_bar, bg_bar)):
            if subcut >= cut:
                S_i = sg_bar[subcut]
                B_i = bg_bar[subcut]
                sig_in_cut += S_i
                bac_in_cut += B_i
        s_above_thresh.append(sig_in_cut)
        b_above_thresh.append(bac_in_cut)

    # calculate metrics:
    for S, B in zip(s_above_thresh, b_above_thresh):
        try:
            median_discovery_significance.append(np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S)))
        except ZeroDivisionError:
            median_discovery_significance.append(0)
        try:
            significance.append(S / np.sqrt(S + B))
        except ZeroDivisionError:
            significance.append(0)
        try:
            purity.append(S / (S + B))
        except ZeroDivisionError:
            purity.append(0)
        try:
            efficiency.append(S / sum(sg_bar))
        except ZeroDivisionError:
            efficiency.append(0)
    return midpoints, sg_bar, bg_bar, median_discovery_significance, significance, purity, efficiency


def analysis(train_labels, test_label):
    os.chdir(sys.path[0])
    params = {'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    with open("config.json") as json_file:
        config = json.load(json_file)


    ########################################################################################################################


    data = prepare_data(pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\Updated_data.pkl"),
                        drop_labels=config['drop_labels'], train_label=train_labels, test_label=test_label)

    train_df = data[data['label'] != -1]

    test_sg = data[data['label'] == -1]
    test_sg['label'] = test_sg['label'].map(lambda foo: 1 if foo == -1 else 0)
    test_bg = train_df[train_df['label'] == 0].sample(frac=0.5)

    # re-normalize Background event weight
    test_bg['EventWeight'] *= 2

    test_df = pd.concat([test_sg, test_bg], ignore_index=True)
    print(len(train_df))
    train_df = train_df.drop(test_bg.index)  # remove test BG events from train_df
    print(len(train_df))

    train_sg = train_df[train_df['label'] == 1]
    train_bg = train_df[train_df['label'] == 0]

    # Generate locally S/B balanced training data
    limits = np.arange(0.3e6, 0.8e6, 0.05e6)
    X_tr, y_tr, w_tr = match_background(train_df, limits, config['variables'], partitions=20, to_df=False)

    # print(len(train_df), len(test_df))
    # print(train_df.head(30))
    # print(test_df.head(30))
    # test_df.to_pickle("./test_df.pkl")
    # train_df.to_pickle("./train_df.pkl")

    xgb = train_model(X_tr, y_tr, w_tr)
    file_name = config["model_filename"]
    pickle.dump(xgb, open(file_name, "wb"))

    X_te, y_te, w_te = rescale(test_df, config['variables'], transform=False)

    print(test_model(xgb, X_te, y_te, w_te))

    midpoints, sg_bar, bg_bar, Med_Disc_Sig, Sig, Pur, Eff = threshold_metrics(xgb, X_te, y_te, w_te,
                                                                               limits=np.linspace
                                                                               (0, 1.05, 21, endpoint=False))
    print(Sig, Pur, Eff)
    plt.figure(figsize=(15, 10))
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

    ax2.plot(midpoints, Med_Disc_Sig)
    ax2.set_ylabel("Median Discovery\n Signal Significance")

    ax3.plot(midpoints, Pur, color="purple")
    ax3.set_ylabel("Signal Purity")
    ax3.set_xlabel("Output threshold")

    eff_colour = "fuchsia"
    ax4.plot(midpoints, Eff, color=eff_colour)
    ax4.tick_params(axis='y', labelcolor=eff_colour)
    ax4.set_ylabel("Signal Efficiency", color=eff_colour)

    ymin, ymax = ax1.get_ylim()
    ax1.axvline(midpoints[Med_Disc_Sig.index(max(Med_Disc_Sig))], ymin=0, ymax=ymax, color='r', ls='-')
    ax2.axvline(midpoints[Med_Disc_Sig.index(max(Med_Disc_Sig))], ymin=0, ymax=1, color='r', ls='-')
    ax3.axvline(midpoints[Med_Disc_Sig.index(max(Med_Disc_Sig))], ymin=0, ymax=1, color='r', ls='-')

    z = np.arange(0, 1.05, 0.05)
    ax1.set_xticks(z)
    ax2.set_xticks(z)
    ax3.set_xticks(z)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    print("Maximum Med. Disc. Sig. of all thresholds: ", max(Med_Disc_Sig))
    plt.show()


    features = config['variables'][:-1]
    print(len(features))
    print(len(xgb.feature_importances_))
    plt.xticks(rotation=90)
    plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
    plt.xticks(np.arange(0, len(features), 1), labels=features)
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.title("Feature Importance: 500GeV test data")
    plt.show()

    X_sg, sg_w,  X_bg, w_bg, y_sg, y_bg = cut_as_filter\
        (fit_model=xgb, X=X_te, w=w_te, chosen_threshold_cut=midpoints[Med_Disc_Sig.index(max(Med_Disc_Sig))], y=y_te)

    _X_sg, _sg_w, _X_bg, _w_bg = filter_unlabelled_data(config=config, threshold=Med_Disc_Sig.index(max(Med_Disc_Sig)))

    plt.figure(figsize=(20, 10))
    plt.title(f"Background like data for a threshold cut of {midpoints[Med_Disc_Sig.index(max(Med_Disc_Sig))]}")

    kwargs = {"histtype": 'step',
              "fill": False}

    fig, ax = plt.subplots(figsize=(15, 10))

    print("Ratio of Signal-l. to Background-l. for Simulated data: ", sum(sg_w) / len(w_bg))
    print("Ratio of Signal-l. to Background-l. for ATLAS 15-16 data: ", sum(_sg_w) / sum(_w_bg))
    ax.hist(_X_bg[:, 2], weights=_w_bg, bins=50, range=[0, 2e6], **kwargs, color='black', label='Background-Like:\nATLAS 2015-16 data')
    ax.hist(X_bg[:, 2], weights=w_bg, bins=50, range=[0, 2e6], **kwargs, color='c', label='Background-like: Simulated')  # 3rd column is mVHres

    # plt.yscale('log')
    ticks = ax.get_xticks()*10**-3  # change from MeV to GeV
    ax.set_xticklabels(ticks)
    plt.legend()
    plt.show()


    FN_cut = (y_sg == 0)
    FN = X_sg[FN_cut]
    FN_w = sg_w[FN_cut]

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title(f"Signal like data for a threshold cut of {midpoints[Med_Disc_Sig.index(max(Med_Disc_Sig))]}")
    ax.hist(_X_sg[:, 2], weights=_sg_w, bins=50, range=[0, 2e6], **kwargs, color='black', normed=True, label='Background-Like:\nATLAS 2015-16 data')
    ax.hist(X_sg[:, 2], weights=sg_w, bins=50, range=[0.2e6, 2e6], **kwargs, facecolor='#F652A0', normed=True, label='True Positive + False Positive')  # 3rd column is mVHres
    #ax.hist(FN[:, 2], weights=FN_w, bins=50, hatch='/', facecolor='c', range=[0.2e6, 2e6], fill=True, normed=False, label='False Positive', edgecolor='k', histtype='step', alpha=0.9)

    ticks = ax.get_xticks()*10**-3  # change from MeV to GeV
    ax.set_xticklabels(ticks)

    ax.set_ylabel("Events")
    ax.set_xlabel("Mass / GeV")
    ax.legend()
    plt.show()


# labels = ('300', '400', '440', '500', '460', '600', '700', '800', '900', '1000', '1400')
# analysis(labels, test_label=[-999])
