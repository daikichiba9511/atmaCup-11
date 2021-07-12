# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
# ---

from pathlib import Path

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# %matplotlib inline

plt.style.use("ggplot")

base_dir = Path("/content/drive/MyDrive/atma/atma_11")
data_dir = base_dir / "input" / "atmacup-11"
print(base_dir)



# +
train = pd.read_csv(data_dir / "train.csv")

display(train.head(10))
display(train.describe())
display(train.info())
# -

data = list(data_dir.glob("*"))
print(data)

sample = pd.read_csv(data_dir / "atmaCup#11_sample_submission.csv")
display(sample.head(10))
sample.shape

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")

import seaborn as sns

test.head(10)

sns.histplot(train, x="target")

train.head(10)

train.shape

# +
print(len(train["object_id"].unique()))

train_uni_rate = len(train["object_id"].unique()) / train.shape[0]
# -

tech = pd.read_csv(data_dir / "techniques.csv")
tech.head(10)

print(tech.shape)
print(len(tech["object_id"].unique()))
tech_unique = len(tech["object_id"].unique()) / tech.shape[0]

mate = pd.read_csv(data_dir / "materials.csv")
mate.head(10)

mate_uni_rate = len(mate["object_id"].unique()) / mate.shape[0]

tech_uni_rate = tech_unique
print(train_uni_rate, tech_uni_rate, mate_uni_rate)
uni_list = [train_uni_rate, tech_uni_rate, mate_uni_rate]
sns.barplot(y=uni_list, x=["train", "technique", "material"])
plt.title("Unique Rate")
plt.show()


