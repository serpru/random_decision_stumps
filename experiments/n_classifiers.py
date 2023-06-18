import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import RDSEnsemble

rnd_seed = 1111

#   Number of splits to check per Stump
n_classifiers = np.linspace(2, 200, 10).astype(int)

data_list = os.listdir(path="../datasets20f")

res = np.zeros((len(n_classifiers), len(data_list), 10))

for k in range(len(data_list)):
    print(f"Working with dataset {k}")
    file_path = "../datasets20f/" + data_list[k]
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
    for j in range(len(n_classifiers)):
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rnd_seed)
        for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
            en = RDSEnsemble(n_classifiers=n_classifiers[j], num_of_splits=65, random_state=rnd_seed)
            en.fit(X[train_index], y[train_index])
            predict = en.predict(X[test_index])
            res[j][k][i] = accuracy_score(y[test_index], predict)

print(res)

np.save(file="../experiment_results/n_classifiers_results", arr=res)
np.save(file="../experiment_results/n_classifiers", arr=n_classifiers)
