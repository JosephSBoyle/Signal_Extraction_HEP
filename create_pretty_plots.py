from flattening import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import atlas_mpl_style as ampl
from data_preprocessing import prepare_data
os.chdir(sys.path[0])
params = {'axes.labelsize' : 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 13,
          'ytick.labelsize': 13}
plt.rcParams.update(params)

with open("config.json") as json_file:
    config = json.load(json_file)

train_labels = ('300', '400', '440', '500', '460', '600', '700', '800', '900', '1000', '1400')
test_label = '10000'
pkl = pd.read_pickle(r"C:\Users\Extasia\Desktop\DISSERTATION\data\Updated_data.pkl")

data = prepare_data(pkl,
                    drop_labels=config['drop_labels'], train_label=train_labels, test_label=test_label)

import seaborn as sn
import mplhep as hep

plt.style.use(hep.style.ROOT)
plt.figure(figsize=(12, 8))
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool), k=1)

cmap = sn.diverging_palette(220, 10, sep=1, as_cmap=True)

cm = sn.heatmap(corr, mask=mask, cmap=cmap,
                annot=False, square=True, center=0,
                annot_kws={'size': 11},
                linewidths=1, linecolor='white')

cm.set_yticklabels(cm.get_yticklabels(), rotation=0)
cm.set_xticklabels(cm.get_xticklabels(), rotation=45)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(r"C:\Users\Extasia\Desktop\DISSERTATION\plots\Sample Correlation Matrix.png", format='png', dpi=1200)

plt.show()

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
limits = np.arange(0.2e6, 1e6, 0.1e6)
X_tr, y_tr, w_tr = match_background(train_df, limits, config['variables'], partitions=10, to_df=False)


# plt.figure(figsize=(25, 25))
# df = pd.DataFrame(np.c_[y_tr, X_tr], columns=["class"] + config['variables'][:-1])
# cm = sn.heatmap(df.corr(), annot=True)
# cm.set_yticklabels(cm.get_yticklabels(), rotation=0)
# cm.set_xticklabels(cm.get_xticklabels(), rotation=90)
# plt.title("Correlation matrix of Features")
# plt.show()

signal_mask = (y_tr == 1)
reversed_mask = np.invert(signal_mask)
signal = X_tr[signal_mask]
signal_w = w_tr[signal_mask]

backg = X_tr[reversed_mask]
backg_w = w_tr[reversed_mask]

######
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[4, 4])

ax1 = plt.subplot(gs[0])

data = prepare_data(pkl, drop_labels=config['drop_labels'], train_label=['500'], test_label=test_label, edit_columns=False)
five_hundred = data[data['label'] == 1]
ax1.hist(five_hundred['mVHres'], weights=five_hundred['EventWeight'], range=[0.4e6, 0.6e6], bins=100)
ticks = ax1.get_xticks() * 10 ** -3  # change from MeV to GeV
ax1.set_xticklabels(ticks)

five_hundred_flat = flatten(five_hundred, 'mVHres', 'EventWeight', 100, manual_lims=[0.45e6, 0.55e6])
ax2 = plt.subplot(gs[1])
ax2.hist(five_hundred_flat['mVHres'], weights=five_hundred_flat['EventWeight'], range=[0.4e6, 0.6e6], bins=100)
ticks = ax2.get_xticks() * 10 ** -3  # change from MeV to GeV
print(len(ax2.get_xticks()))
print(len(ticks))
ax2.set_xticklabels(ticks)
# ax1 = plt.subplot(gs[0])
# train_labels = ('500')
#
# ax1.hist(five_hundred_sg['mVHres'], weights=five_hundred_sg['EventWeight'], range=[0.3e6, 0.7e6])
#
# ax2 = plt.subplot(gs[1])
# ax2.hist(five_hundred_sg['mVHres'], weights=five_hundred_sg['EventWeight'], range=[0.3e6, 0.7e6])

ax3 = plt.subplot(gs[2])
ax3.hist(train_bg['mVHres'], color='red', weights=train_bg['EventWeight'], range=[0.2e6, 1e6], normed=True, bins=100, label='background')
ax3.hist(train_sg['mVHres'], weights=train_sg['EventWeight'], range=[0.2e6, 1e6], normed=True, bins=100, alpha=0.75)


ax4 = plt.subplot(gs[3])

ax4.hist(backg.T[2], histtype='step', range=[0.2e6, 1e6], weights=backg_w, color='red', bins=50, linewidth=2.5,
         fill=True)

N, bins, patches = ax4.hist(signal.T[2], range=[0.2e6, 1e6], weights=signal_w, bins=50, color='series:pink',
                            alpha=0.8)

for it, j in enumerate(bins[:-2]):
    print(it)
    if j > 0.3e6:
        patches[it].set_facecolor('series:cyan')
        if j > 0.4e6:
            patches[it].set_facecolor('series:orange')
            if j > 0.5e6:
                patches[it].set_facecolor('series:cyan')
                if j > 0.6e6:
                    patches[it].set_facecolor('series:blue')
                    if j > 0.7e6:
                        patches[it].set_facecolor('series:green')
                        if j > 0.8e6:
                            patches[it].set_facecolor('series:purple')
                            if j > 0.9e6:
                                patches[it].set_facecolor('series:pink')

ticks = ax3.get_xticks() * 10 ** -3  # change from MeV to GeV
ax3.set_xticklabels(ticks)
ax3.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

ticks = ax4.get_xticks() * 10 ** -3  # change from MeV to GeV
ax4.set_xticklabels(ticks)

ax1.set_title("500 GeV Signal simulation")
ax1.set_xlabel("m(VH) / GeV")
ax1.set_ylabel("Weighted Events")

ax2.set_title("After flattening the Signal:")
ax2.set_xlabel("m(VH) / GeV")
ax2.set_ylabel("Weighted Events")

ax3.set_title("Simulated Signal and Background")
ax3.set_xlabel("m(VH) / GeV")
ax3.set_ylabel("Weighted Events\nNormalised for visual effect")

ax4.set_title("Signal Locally Flattened to match Background")
ax4.set_xlabel("m(VH) / GeV")
ax4.set_ylabel("Weighted Events")
gs.tight_layout(fig)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
fig.savefig(r"C:\Users\Extasia\Desktop\DISSERTATION\plots\DFA.png", format='png', dpi=1200)
plt.show()
