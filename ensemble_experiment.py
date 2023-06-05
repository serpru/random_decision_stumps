import numpy as np
from numpy.random.mtrand import random
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from Random_Decision_Stumps import OurRDS
from Random_Decision_Stumps import RDSEnsemble

import matplotlib.pyplot as plot

rnd = 1111

features = 20
n_classifiers = 10

#   TODO pobierac dane z pliku
x, y = datasets.make_classification(
    n_samples=400,
    n_features=features,
    n_informative=features,
    n_repeated=0,
    n_redundant=0,
    random_state=rnd,
)

ensemble_a_score = []
prediction = []
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rnd)
for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
    ensemble = RDSEnsemble(random_state=rnd, num_of_splits=100, n_classifiers=n_classifiers)
    ensemble.fit(x[train_index], y[train_index])
    ensemble_pred = ensemble.predict(x[test_index])
    prediction.append(ensemble_pred)
    ensemble_a_score.append(accuracy_score(y[test_index], ensemble_pred))


ensemble_mean = np.mean(ensemble_a_score)
ensemble_std = np.std(ensemble_a_score)

print(f"MyClassificator mean accuracy score: {round(ensemble_mean, 3)}")
print(f"MyClassificator std score: {round(ensemble_std, 3)}")


# Single Ensemble Experiment + plot
# x_learn, x_test, y_learn, y_test = train_test_split(
#     x,
#     y,
#     test_size=0.2,
#     train_size=0.8,
#     random_state=rnd,
# )
#
# ensemble = RDSEnsemble(random_state=rnd, num_of_splits=100, n_classifiers=n_classifiers)
# ensemble.fit(x_learn, y_learn)
#
# predict_from_learn = ensemble.predict(x_learn)
#
# prediction = ensemble.predict(x_test)
#
# acc = accuracy_score(y_test, prediction)
#
# it = 0
# for classifier in ensemble.classifiers:
#     print(f"Classifier {it} direction")
#     print(classifier.best_direction)
#     print(classifier.best_split_point)
#     it += 1
#
#
# fig, ax = plot.subplots(
#     nrows=2,
#     ncols=2,
#     sharex=True,
#     sharey=True,
# )
#
#
#
# ax[0][0].scatter(x_learn[:, 0], x_learn[:, 1], c=y_learn, cmap="coolwarm")
# fig.suptitle(f"Experiment: single RDS ensemble. Accuracy score: {acc}\nNum of classifiers {n_classifiers} Seed {rnd} ")
# ax[0][0].set_title("Real data")
#
# ax[1][0].scatter(x_learn[:, 0], x_learn[:, 1], c=predict_from_learn, cmap="coolwarm")
# ax[1][0].axvline(x=ensemble.classifiers[0].best_split_point, color='black', linestyle="dashed", alpha=0.7,
#                           label='axvline - full height')
# ax[1][0].axhline(y=ensemble.classifiers[1].best_split_point, color='black', linestyle="dashed", alpha=0.7,
#                           label='axvline - full height')
# ax[1][0].set_title("Prediction from learn")
#
# ax[0][1].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm")
# ax[0][1].set_title("Test data")
#
# ax[1][1].scatter(x_test[:, 0], x_test[:, 1], c=prediction, cmap="coolwarm")
# ax[1][1].set_title("Prediction")
#
# plot.show()