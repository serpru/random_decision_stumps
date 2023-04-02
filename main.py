import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    print("Projekt na MSI")

    x, y = datasets.make_classification(
        weights=None,
        n_samples=400,
        n_features=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        flip_y=0.08,
        random_state=1,
        n_clusters_per_class=2
    )

    x_learn, x_test, y_learn, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=1,
    )

    # Random Forest
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    rfc.fit(x_learn, y_learn)

    rfc_pred = rfc.predict(x_test)


    # Decision Tree
    dtc = DecisionTreeClassifier(max_depth=2, random_state=0)
    dtc.fit(x_learn, y_learn)

    dtc_pred = dtc.predict(x_test)


    print(f"real:\n{y_test}")
    print(f"rfc_pred:\n{rfc_pred}")
    print(f"dtc_pred:\n{dtc_pred}")