import pandas as pd
from data_preprocessing import *
from filter import cut_as_filter
import pickle
import matplotlib.patches as patches


def filter_unlabelled_data(config, threshold, data, model):


    print(data.label.isnull().sum())  # ensure data isn't sparse!!!
    assert data["EventWeight"].unique()[0] == 1.0


    X, y, w = rescale(data, config['variables'], transform=False)

    X_sg, sg_w,  X_bg, w_bg = cut_as_filter\
        (fit_model=xgb_model_loaded, X=X, w=w, chosen_threshold_cut=threshold, y=None)
    return X_sg, sg_w,  X_bg, w_bg


def plot_threshold(trained_model, data, thresh, bins=20):
    limits = np.linspace(0, 1, 10, endpoint=False)
    figure, ax = plt.subplots()
    y = []
    for cut in limits:
        cut_results = filter_unlabelled_data(config=config, threshold=cut, data=data, model=trained_model)
        y.append(len(cut_results[1]))

    yerr = [np.sqrt(i) for i in y]
    ax.errorbar(limits, y, yerr=yerr, color="black", fmt='o')
    ax.set_yscale('log')
    plt.ylabel("Events")
    plt.xlabel("Threshold")
    rect = patches.Rectangle((thresh, 1), 1 - thresh, max(y) - min(y), facecolor='grey',
                             transform=figure.transFigure)

    ax.add_patch(
        patches.Rectangle((thresh, 0), 0.2, max(y), zorder=5))


    ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    # Load data
    with open("config.json") as json_file:
        config = json.load(json_file)


    CERN_data = pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\CERN_2015-16.pkl").loc["data", :]
    CERN_data = prepare_data(CERN_data, drop_labels=config['drop_labels'], train_label=['NONE'],
                                 test_label="data")
    # Load Xgb model
    file_name = config["model_filename"]
    xgb_model_loaded = pickle.load(open(file_name, "rb"))
    print(xgb_model_loaded)


    plot_threshold(data=CERN_data, thresh=0.8, bins=10, trained_model=xgb_model_loaded)

    X_sg, sg_w,  X_bg, w_bg = filter_unlabelled_data(config=config, threshold=0.80, data=CERN_data, model=xgb_model_loaded)

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title(f"Background like data for a threshold cut of {0.875}")
    ax.hist(X_bg[:, 2], weights=w_bg, bins=50, range=[0, 2e6], edgecolor='black')  # 3rd column is mVHres
    ticks = ax.get_xticks()*10**-3  # change from MeV to GeV
    ax.set_xticklabels(ticks)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title(f"Signal like data for a threshold cut of {0.875}")
    ax.hist(X_sg[:, 2], weights=sg_w, bins=50, range=[0.2e6, 2e6], edgecolor='black',
            facecolor='#F652A0', normed=False, label='True Positive + False Positive')  # 3rd column is mVHres

    ticks = ax.get_xticks()*10**-3  # change from MeV to GeV
    ax.set_xticklabels(ticks)

    ax.set_ylabel("Events")
    ax.set_xlabel("Mass / GeV")
    ax.legend()
    plt.show()
