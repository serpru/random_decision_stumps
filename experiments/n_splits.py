import os
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import DecisionStump
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

rnd_state = 1111

#   Number of splits to check per Stump
num_of_splits = np.linspace(5, 500, 50).astype(int)

data_list = os.listdir(path="../datasets20f")

res = np.zeros((len(num_of_splits), len(data_list), 10))

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
    for j in range(len(num_of_splits)):
        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rnd_state)
        for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
            ds = DecisionStump(random_state=rnd_state, num_of_splits=num_of_splits[j])
            ds.fit(X[train_index, 0], y[train_index])
            predict = ds.predict(X[test_index, 0])
            res[j][k][i] = accuracy_score(y[test_index], predict)

print(res)

np.save(file="../experiment_results/n_split_results", arr=res)
np.save(file="../experiment_results/n_split_ranges", arr=num_of_splits)
