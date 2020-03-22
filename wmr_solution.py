#  Create dataset which contains simul masses as labels for signal events and 0 for ttbar / Z background.
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import prepare_data, background_split, rescale
from model import train_model, test_model
from flattening import match_background


def simul_label(label, _, __):
    if label in ("ttbar", "Zlljet", "zlljet"):
        return 0
    else:
        return int(label)


def binary_label(label):
    if label in ("ttbar", "Zlljet", "zlljet"):
        return 0
    else:
        return 1


def split(df, targ_w):
    """
    Similar to sklearn's train_test_split, but instead of test frac, aims for a specifc event weight in the test set
    :param df:
    :param targ_w:
    :return:
    """
    te_sg = None
    print([simul_label(i, 0, 0) for i in ('1000', '1400', '300', '400', '440', '460', '500', '600', '700', '800', '900')])
    df['label'] = df['label'].map(lambda x: simul_label(x, 0, 0))
    print(df.label.unique())
    print(df.columns)

    for i in ('1000', '1400', '300', '400', '440', '460', '500', '600', '700', '800', '900'):
        sub_df = df[df['label'] == int(i)]
        n = len(sub_df)
        NEW = sub_df.sum()['EventWeight']
        mean_w = NEW / n  # avg. sample weight
        m = int(targ_w / mean_w)  # events to subsample

        subsample = sub_df.sample(m)
        if te_sg is None:
            te_sg = subsample
        else:
            te_sg = pd.concat([te_sg, subsample], ignore_index=True)

        print(te_sg.sum()['EventWeight'])
        tr_sg = df.drop(te_sg.index)
        tr_sg['label'] = tr_sg['label'].map(binary_label)

    te_sg['label'] = np.ones(len(te_sg))

    background_df = df[df['label'] == 0]
    tr_bg,  te_bg = background_split(background_df)

    test_df = pd.concat([te_bg, te_sg])
    train_df = pd.concat([tr_bg, tr_sg])

    test_df['label'] = test_df['label'].astype(int)
    print("NOTE POSITION OF mVHres!!!", train_df.columns)
    return train_df.drop(columns=[b'label']), test_df.drop(columns=[b'label'])


with open("config.json") as json_file:
    config = json.load(json_file)

# labels are given because they are required arguments, they do not alter any behaviour.
data = prepare_data(pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\Updated_data.pkl"),
                    drop_labels=config['drop_labels'], train_label=0, test_label=0,
                    relabelling_func=simul_label)


train_df, test_df = split(data, targ_w=50)
print(train_df.columns)
print("\n\n\n", train_df.label.unique(), test_df.label.unique(), "\n\n\n")

limits = np.linspace(0.3e6, 1e6, 20)
X_tr, y_tr, w_tr = match_background(train_df, limits=limits, variables=config['variables'], partitions=5)

X_te, y_te, w_te = rescale(test_df, config['variables'], transform=False)

# remove events outside the training mass limits
upper_mass_lim = max(limits)
lower_mass_lim = min(limits)
mask = [np.logical_and(X_te[:,2] > lower_mass_lim, X_te[:,2] < upper_mass_lim)]
print(len(y_te))
X_te = X_te[mask]
y_te = y_te[mask]
w_te = w_te[mask]
print(len(y_te))
print(np.shape(X_tr), np.shape(y_tr), np.shape(w_tr))
print(np.shape(X_te), np.shape(y_te), np.shape(w_te))
print(type(X_tr), "\n\n\n")
# MODELLING

# UNI-MODEL APPROACH
mva = train_model(X_tr, y_tr, w_tr)
results = test_model(mva, X_te, y_te, w_te)
print("UNI-MODEL RESULTS: \n", results)

ax = plt.figure(figsize=(18, 14))
features = config['variables'][:-1]
print(len(features))
print(len(mva.feature_importances_))
plt.xticks(rotation=90)
plt.bar(range(len(mva.feature_importances_)), mva.feature_importances_)
plt.xticks(np.arange(0, len(features), 1), labels=features)
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.title("Feature Importance: Merged Signal")
plt.show()


# train n mva models, corresponding to n mass bins. Store these models in a dictionary with keys of form:
#  mva_lowerMassLimit_upperMassLimit
def train_binned_models(arrays, min_max, n_bins):
    models = {}
    X, y, w = arrays
    print(min(X[:,2]), max(X[:,2]))
    bin_lims = np.linspace(*min_max, n_bins+1)  # n bins requires n+1 limits.
    print(bin_lims)

    # data in [ll, ul)
    for i in range(len(bin_lims) - 1):
        ll, ul = bin_lims[i], bin_lims[i + 1]
        print(ll, ul)

        # mask: lower limit <= mVH < upper limit
        # NB ensure mVHres is in fact the 3rd column of X!!! [:,2]
        mask = (X[:, 2] > ll) & (X[:, 2] < ul)
        bin_X = X[mask]
        bin_y = y[mask]
        bin_w = w[mask]
        print(bin_X, bin_y)
        models[f"mva_{ll}_{ul}"] = train_model(bin_X, bin_y, bin_w)

        print(test_model(models[f"mva_{ll}_{ul}"], bin_X, bin_y, bin_w))

    print(models.keys())
    print(bin_lims)
    return models, bin_lims


def test_binned_models(models, bin_lims, arrays):
    X, y, w = arrays
    print(np.shape(X), np.shape(y))

    def map_model(sample, models):
        mVH = sample[2]
        sample = np.array(sample).reshape((1, -1))
        for i in range(len(bin_lims) - 1):
            ll = bin_lims[i]
            ul = bin_lims[i+1]
            if ll <= mVH <= ul:
                return models[f"mva_{ll}_{ul}"].predict(sample)
        raise ValueError

    ypred = np.zeros(len(y))
    for i in range(len(y)):
        ypred[i] = map_model(X[i, :], models)
    # ypred = np.apply_along_axis(func1d=map_model, axis=1, arr=X, models=models)
    print(ypred)
    print(X[:,2])
    print(np.shape(ypred))
    print(np.mean(ypred))
    return ypred


mvas, limz = train_binned_models([X_tr, y_tr, w_tr], min_max=[0.3e6, 1e6], n_bins=7)
print(limz)
ypred = test_binned_models(mvas, limz, [X_te, y_te, w_te])

####
signal_sum = 0
backg_sum = 0
for c, (i, j) in enumerate(zip(ypred, y_te)):
    if i == 1:
        if j == 1:
            signal_sum += w_te[c]
        else:
            backg_sum += w_te[c]

try:
    purity = signal_sum / (backg_sum + signal_sum)
    significance = signal_sum / np.sqrt(backg_sum + signal_sum)
except ZeroDivisionError:
    purity = 0
    significance = 0
results = {'FoM: S / sqrt(S + B)': significance, 'FoM: S / (S + B)': purity,
                                         'Signal weight "TP"': signal_sum, 'Background weight "FP"': backg_sum}

print(results)
# @ TODO optimize FoM????
# create l1 pre-classifier
# test 2 stage classifier with optimized output threshold (max(significance)) - needs to be tuned for each bin vs
# uni-classifier with optimized output threshold (max(significane)
