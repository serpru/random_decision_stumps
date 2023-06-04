import random
import os
import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import OurRDS, DecisionStump, RDSEnsemble

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


rnd_state = 1111

data_list = os.listdir(path="./datasets")

classifiers = [GaussianNB(), DecisionTreeClassifier(max_depth=1, random_state=rnd_state), DecisionStump(random_state=rnd_state, num_of_splits=50)]

res = np.zeros((len(data_list), 10, len(classifiers)))

for j in range(len(data_list)):

    file_path = "./datasets/" + data_list[j]
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
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rnd_state)
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        for k in range(len(classifiers)):
            classifiers[k].fit(X[train_index], y[train_index])
            predict = classifiers[k].predict(X[test_index])
            res[j][i][k] = accuracy_score(y[test_index], predict)

print("Print res")
print(res)
print("Print res shape")
print(res.shape)


#np.save(file="results", arr=res)
