import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from Random_Decision_Stumps import OurRDS

if __name__ == '__main__':
    print("Projekt na MSI")

    # Seed
    rnd_seed = 1

    # Dataset
    x, y = datasets.make_classification(
        weights=None,
        n_samples=400,
        n_features=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        flip_y=0.08,
        random_state=rnd_seed,
        n_clusters_per_class=2
    )

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
        n_repeats=5,
        random_state=rnd_seed,
    )

    # Our Random Decision Stump
    ds = OurRDS()

    # Random Forest
    a_score_rfc = []

    for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
        rfc = RandomForestClassifier(max_depth=2, random_state=rnd_seed, criterion="gini")
        rfc.fit(x[train_index], y[train_index])
        predict = rfc.predict(x[test_index])
        a_score_rfc.append(accuracy_score(y[test_index], predict))

    # Decision Tree
    a_score_dtc = []

    for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
        dtc = DecisionTreeClassifier(max_depth=2, random_state=rnd_seed, criterion="gini")
        dtc.fit(x[train_index], y[train_index])
        predict = dtc.predict(x[test_index])
        a_score_dtc.append(accuracy_score(y[test_index], predict))



    # Random temporary stuff for debug, delete later
    print(f"real:\n{y_test}")

    print(f"random forest acc score after folds:\n" "%.3f" % np.mean(a_score_rfc))
    print(f"decision tree acc score after folds:\n" "%.3f" % np.mean(a_score_dtc))
