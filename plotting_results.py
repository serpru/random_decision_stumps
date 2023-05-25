import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from Gini import Gini
from Random_Decision_Stumps import OurRDS


# Dataset
x = np.load(file="x.npy")
y = np.load(file="y.npy")

gini_data = np.load("gini_data.npy")
split_data = np.load("split_data.npy")


#   Plotting single experiment
fig, ax = plot.subplots(
    nrows=2,
    ncols=3,
    sharex=True,
    sharey=True,
)

cos = []
for i in split_data[0]:
    cos.append(1)

ax[0][0].scatter(split_data[0], gini_data[0], c=cos, cmap="coolwarm")
fig.suptitle(f"Experiment - Gini(num_of_split) per Stump ")
ax[0][0].set_title("5 splits per Stump")

ax[1][0].scatter(split_data[1], gini_data[1], c=cos, cmap="coolwarm")
ax[1][0].set_title("20 splits per Stump")

ax[0][1].scatter(split_data[2], gini_data[2], c=cos, cmap="coolwarm")
ax[0][1].set_title("50 splits per Stump")

ax[1][1].scatter(split_data[3], gini_data[3], c=cos, cmap="coolwarm")
ax[1][1].set_title("200 splits per Stump")

ax[0][2].scatter(split_data[4], gini_data[4], c=cos, cmap="coolwarm")
ax[0][2].set_title("500 splits per Stump")

ax[1][2].scatter(split_data[5], gini_data[5], c=cos, cmap="coolwarm")
ax[1][2].set_title("2000 splits per Stump")

plot.show()


