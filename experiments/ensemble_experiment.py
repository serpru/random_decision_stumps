import random
import os
import numpy as np
from sklearn.model_selection import  RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import RDSEnsemble

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


rnd_seed = 1111
n_classifiers = 175
n_splits = 65

data_list = os.listdir(path="../datasets20f")

classifiers = [
    GaussianNB(),
    DecisionTreeClassifier(max_depth=1, random_state=rnd_seed),
    RDSEnsemble(n_classifiers=n_classifiers, random_state=rnd_seed, num_of_splits=n_splits)
]

res = np.zeros((len(data_list), 10, len(classifiers)))

for j in range(len(data_list)):

    file_path = "../datasets20f/" + data_list[j]
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

    a_score = []
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rnd_seed)
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        for k in range(len(classifiers)):
            classifiers[k].fit(X[train_index], y[train_index])
            predict = classifiers[k].predict(X[test_index])
            res[j][i][k] = accuracy_score(y[test_index], predict)

print("Print res")
print(res)
print("Print res shape")
print(res.shape)

exit()
np.save(file="../experiment_results/ensemble_results", arr=res)
