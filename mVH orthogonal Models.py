from flattening import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import atlas_mpl_style as ampl
import mplhep as hep
from data_preprocessing import prepare_data
from threshold_analysis import threshold_metrics


def mVH_orthogonal_analysis(dataframe, variables, train_label, test_label):
    data = prepare_data(dataframe, drop_labels=[3, 4], train_label=train_label, test_label=test_label)

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

    # boost training signal weight to match background (train)
    boost_frac = sum(train_df[train_df['label'] == 0]['EventWeight']) / sum(train_df[train_df['label'] == 1]['EventWeight'])

    mask = (train_df['label'] == 1)
    tr_ = train_df[mask]
    train_df.loc[mask, 'EventWeight'] = tr_['EventWeight'] * boost_frac

    # variables *must* have have EventWeight as the last value.
    # variables = data.columns[1:-1]

    Xtrain, ytrain, tr_w = rescale(train_df, variables)
    Xtest, ytest, te_w = rescale(test_df, variables)

    print(shape(Xtrain))
    xgb = train_model(Xtrain, ytrain, tr_w)

    midpoints, sg_bar, bg_bar, Med_Disc_Sig, Sig, Pur, Eff = threshold_metrics(xgb, Xtest, ytest, te_w,
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

    plt.style.use(hep.style.ROOT)
    ax1.set_title(f"Hold-out test set of {test_label} GeV\nBest Median Significance: {max(Med_Disc_Sig)}")
    plt.savefig(f"C:\\Users\\Extasia\\Desktop\\DISSERTATION\\plots\\Naive Method\\{test_label}_{max(Med_Disc_Sig)}.png",
                format='png', dpi=400)
    # plt.show()


DATA = pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\Updated_data.pkl")

params = {'axes.labelsize' : 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 15,
          'ytick.labelsize': 15}
plt.rcParams.update(params)

train_label = ['300', '400', '440', '460', '500', '600', '700', '800', '900', '1000', '1400']

variables = ['l1px', 'l1py', 'l1pz', 'l2px', 'l2py', 'l2pz', 'METHT',
             'mLL', 'dEtaLL',
             'EventWeight']
variables = ["mBBres", "pTV",
    "mVHres", "mLL", "METHT", "dEtaLL", "dPhiLL",
       "ptL1", "ptL2", "j1e", "j1px", "j1py", "j2e", "j1pz", "j2px", "j2py",
       "j2pz", "l1e", "l1px", "l1py", "l1pz", "l2e", "l2px", "l2py", "l2pz",
       "EventWeight"]

for i in train_label:
    train_label_ = [j for j in train_label if j != i]
    assert i not in train_label_
    mVH_orthogonal_analysis(DATA, variables, train_label=train_label_, test_label=i)
