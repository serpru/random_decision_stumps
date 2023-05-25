import random

import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import OurRDS


# Random Seed
rnd_seed = 3423
# 12442
# 9865

#   Number of splits to check per Stump
num_of_splits = [5, 20, 50, 200, 500, 2000]

# Dataset
x = np.load(file="x.npy")
y = np.load(file="y.npy")

# Split train/test
x_learn, x_test, y_learn, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    train_size=0.8,
    random_state=rnd_seed,
)

# Folds
rskf = RepeatedStratifiedKFold(
    n_splits=2,
    n_repeats=30,
    random_state=rnd_seed,
)

gini_data = []
split_data = []

a_score_ords = []

for j in range(len(num_of_splits)):
    a_score_ords.append([])
    gini_data.append([])
    split_data.append([])
    for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
        ords = OurRDS(num_of_splits=num_of_splits[j], random_state=rnd_seed)
        ords.fit(x[train_index], y[train_index])
        predict_ords = ords.predict(x[test_index])
        gini_data[j].append(ords.best_gini)
        split_data[j].append(ords.best_split_point)
        a_score_ords[j].append(accuracy_score(y[test_index], predict_ords))

a_score_np = np.array(a_score_ords)

print("Results")
print(a_score_np)

np.save(file="n_split_results", arr=a_score_np)

#   Plotting single experiment
fig, ax = plot.subplots(
    nrows=2,
    ncols=3,
    sharex=True,
    sharey=True,
)

gdata = np.array(gini_data)
spdata = np.array(split_data)

np.save(file="gini_data", arr=gdata)
np.save(file="split_data", arr=spdata)

cos = []
for i in split_data[0]:
    cos.append(1)

ax[0][0].scatter(split_data[0], gini_data[0], c=cos, cmap="coolwarm")
fig.suptitle(f"Experiment - number of splits per stump. Repeated Stratified K fold, 2 splits, 30 repeats ")
ax[0][0].set_title(f"Best gini values for {num_of_splits[0]} splits per Stump")

ax[1][0].scatter(split_data[1], gini_data[1], c=cos, cmap="coolwarm")
ax[1][0].set_title(f"Best gini values for {num_of_splits[1]} splits per Stump")

ax[0][1].scatter(split_data[2], gini_data[2], c=cos, cmap="coolwarm")
ax[0][1].set_title(f"Best gini values for {num_of_splits[2]} splits per Stump")

ax[1][1].scatter(split_data[3], gini_data[3], c=cos, cmap="coolwarm")
ax[1][1].set_title(f"Best gini values for {num_of_splits[3]} splits per Stump")

ax[0][2].scatter(split_data[4], gini_data[4], c=cos, cmap="coolwarm")
ax[0][2].set_title(f"Best gini values for {num_of_splits[4]} splits per Stump")

ax[1][2].scatter(split_data[5], gini_data[5], c=cos, cmap="coolwarm")
ax[1][2].set_title(f"Best gini values for {num_of_splits[5]} splits per Stump")

plot.show()