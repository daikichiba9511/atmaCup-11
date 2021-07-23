# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3.7.11 64-bit
#     name: python3
# ---

# +
import pandas as pd
import os

os.chdir("/content/drive/MyDrive/atma/atma_11")
sub_000 = pd.read_csv("./output/exp000/submission.csv")
sub_004 = pd.read_csv("./output/exp004/submission.csv")
# -

sub_000.head()

sub_004.head()

sub_005 = pd.read_csv("./output/exp005/submission.csv")
sub_007 = pd.read_csv("./output/exp007/submission.csv")
# sub_008 = pd.read_csv("./output/exp008/submission.csv")
sub_009 = pd.read_csv("./output/exp009/submission.csv")
sub_013 = pd.read_csv("./output/exp013/submission.csv")
sub_014 = pd.read_csv("./output/exp014/submission.csv")
sub_016 = pd.read_csv("./output/exp016/submission.csv")
sub_020 = pd.read_csv("./output/exp020/submission.csv")
sub_024 = pd.read_csv("./output/exp024/submission.csv")


sub_016.shape

sub_020.shape

# +
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use("ggplot")
plt.hist(sub_000["target"].values, label="exp000", alpha=0.4)
# plt.hist(sub_004["target"].values, label="exp004", alpha=0.4)
plt.hist(sub_005["target"].values, label="exp005", alpha=0.4)
# plt.hist(sub_007["target"].values, label="exp007", alpha=0.4)
# plt.hist(sub_008["target"].values, label="exp008", alpha=0.4)
# plt.hist(sub_009["target"].values, label="exp009", alpha=0.4)
plt.hist(sub_013["target"].values, label="exp013", alpha=0.4)
plt.hist(sub_014["target"].values, label="exp014", alpha=0.4)
plt.hist(sub_016["target"].values, label="exp016", alpha=0.4)
plt.hist(sub_020["target"].values, label="exp020", alpha=0.4)
plt.hist(sub_024["target"].values, label="exp024", alpha=0.4)

plt.legend()
plt.show()

# +
from IPython.display import display

display(sub_000.describe())
display(sub_005.describe())
display(sub_013.describe())
display(sub_016.describe())
# -

train = pd.read_csv("input/atmacup-11/train.csv")



train


