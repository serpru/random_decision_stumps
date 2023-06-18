import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import RDSEnsemble
import matplotlib.pyplot as plot

rnd_seed = 1111
num_of_splits = 65
n_classifiers = 2

data_list = os.listdir(path="../datasets20f")

file_path = "../datasets20f/" + data_list[0]
data = np.loadtxt(file_path, delimiter=",", dtype=str)

X = []
for item in data:
    X.append(item[:-1])

X = np.array(X)
X = X.astype(dtype=float)

y = []
for item in data:
    y.append(item[-1])
y = np.array(y)
y = y.astype(dtype=int)

# Split train/test
x_learn, x_test, y_learn, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    train_size=0.8,
    random_state=rnd_seed,
)
en = RDSEnsemble(n_classifiers=n_classifiers, num_of_splits=num_of_splits, random_state=rnd_seed)
en.fit(x_learn, y_learn)
pred = en.predict(x_test)

print(en.x_indexes)

learned_y = en.predict(x_learn)

learned_y_class0 = en.classifiers[0].predict(x_learn[:, en.x_indexes[0]])
learned_y_class1 = en.classifiers[1].predict(x_learn[:, en.x_indexes[1]])

pred_class0 = en.classifiers[0].predict(x_test[:, en.x_indexes[0]])
pred_class1 = en.classifiers[1].predict(x_test[:, en.x_indexes[1]])

acc_class0 = accuracy_score(pred_class0, y_test)
acc_class1 = accuracy_score(pred_class1, y_test)

learned_y_class0[learned_y_class0 == 0] = -1
learned_y_class1[learned_y_class1 == 0] = -1

cmap_weighted = np.array(learned_y_class0)*en.weights[0] + np.array(learned_y_class1)*en.weights[1]


a_score = accuracy_score(y_test, pred)

fig, ax = plot.subplots(
    nrows=2,
    ncols=3,
    sharex=True,
    sharey=True,
)

ax[0][0].scatter(x_learn[:, en.x_indexes[0]], x_learn[:, en.x_indexes[1]], c=y_learn, cmap="coolwarm")
fig.suptitle(f"Experiment: single RDS ensemble. Accuracy score: {a_score}\nNum of classifiers {n_classifiers} Seed {rnd_seed} ")
ax[0][0].set_title("Real data")

ax[1][0].scatter(x_learn[:, en.x_indexes[0]], x_learn[:, en.x_indexes[1]], c=cmap_weighted, cmap="coolwarm")
ax[1][0].axvline(x=en.classifiers[0].best_split_point, color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')
ax[1][0].axhline(y=en.classifiers[1].best_split_point, color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')
ax[1][0].set_title("Learned split")

ax[0][1].scatter(x_learn[:, en.x_indexes[0]], x_learn[:, en.x_indexes[1]], c=learned_y_class0, cmap="coolwarm")
ax[0][1].set_title(f"Learned split for classifier 0\nAccuracy score against test data {acc_class0}")
ax[0][1].axvline(x=en.classifiers[0].best_split_point, color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')

ax[1][1].scatter(x_learn[:, en.x_indexes[0]], x_learn[:, en.x_indexes[1]], c=learned_y_class1, cmap="coolwarm")
ax[1][1].set_title(f"Learned split for classifier 1\nAccuracy score against test data {acc_class1}")
ax[1][1].axhline(y=en.classifiers[1].best_split_point, color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')

ax[0][2].scatter(x_test[:, en.x_indexes[0]], x_test[:, en.x_indexes[1]], c=y_test, cmap="coolwarm")
ax[0][2].set_title("Test data")

ax[1][2].scatter(x_test[:, en.x_indexes[0]], x_test[:, en.x_indexes[1]], c=pred, cmap="coolwarm")
ax[1][2].axvline(x=en.classifiers[0].best_split_point, color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')
ax[1][2].axhline(y=en.classifiers[1].best_split_point, color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')
ax[1][2].set_title("Prediction")

plot.show()
