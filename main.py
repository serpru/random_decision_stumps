import random

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

if __name__ == '__main__':
    print("Projekt na MSI")

    # Seed
    rnd_seed = 12442

    # Dataset
    x, y = datasets.make_classification(
        weights=None,
        n_samples=300,
        n_features=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        flip_y=0.08,
        random_state=rnd_seed,
        n_clusters_per_class=1
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
    split_point = 0.5
    ds = OurRDS(split_point=split_point, num_of_splits=20, random_state=rnd_seed)

    # Random Forest
    # a_score_rfc = []
    #
    # for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
    #     rfc = RandomForestClassifier(max_depth=2, random_state=rnd_seed, criterion="gini")
    #     rfc.fit(x[train_index], y[train_index])
    #     predict = rfc.predict(x[test_index])
    #     a_score_rfc.append(accuracy_score(y[test_index], predict))

    # Decision Tree
    # a_score_dtc = []
    #
    # for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
    #     dtc = DecisionTreeClassifier(max_depth=2, random_state=rnd_seed, criterion="gini")
    #     dtc.fit(x[train_index], y[train_index])
    #     predict = dtc.predict(x[test_index])
    #     a_score_dtc.append(accuracy_score(y[test_index], predict))



    # Random temporary stuff for debug, delete later
    # print(f"real:\n{y_test}")
    #
    # print(f"random forest acc score after folds:\n" "%.3f" % np.mean(a_score_rfc))
    # print(f"decision tree acc score after folds:\n" "%.3f" % np.mean(a_score_dtc))
    #
    #
    # print(y_test)
    # print(x_test)

    new_x_learn = []
    new_x_test = []
    for i in range(len(x_learn)):
        new_x_learn.append([x_learn[i][0]])

    for i in range(len(x_test)):
        new_x_test.append([x_learn[i][0]])

    print(new_x_learn)
    print(x_learn)

    ds.fit(new_x_learn, y_learn)

    ds_learn_y = ds.predict(new_x_learn)
    ds_predict = ds.predict(new_x_test)

    print("Best Gini")
    print(ds.best_gini)
    print(ds.gini_list)
    print("Best split point")
    print(ds.best_split_point)

    # plot.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="coolwarm")
    # plot.axvline(x=split_point, color='b', label='axvline - full height')
    # plot.xlabel("Atrybut 1")
    # plot.ylabel("Atrybut 2")
    # plot.title("Zadanie 1")
    # plot.show()
    #
    # plot.scatter(x_test[:, 0], x_test[:, 1], c=predict_ds, cmap="coolwarm")
    # plot.axvline(x=split_point, color='b', label='axvline - full height')
    # plot.xlabel("Atrybut 1")
    # plot.ylabel("Atrybut 2")
    # plot.title("Zadanie 1")
    # plot.show()

    fig, ax = plot.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
    )
    random.seed(rnd_seed)
    arr_learn = []
    for x in x_learn:
        n = random.triangular(0.1, 0.9)
        arr_learn.append(0.5 + n)

    arr_test = []
    for x in x_test:
        n = random.triangular(0.1, 0.9)
        arr_test.append(0.5 + n)

    a_score = accuracy_score(y_test, ds_predict)
    print(f"Accuracy score: {a_score}")

    ax[0][0].scatter(new_x_learn[:], arr_learn, c=y_learn, cmap="coolwarm")
    fig.suptitle("Tytul okna")
    ax[0][0].set_title("Learning data")

    ax[1][0].scatter(new_x_learn[:], arr_learn, c=ds_learn_y, cmap="coolwarm")
    for i in range(len(ds.gini_list)):
        if (ds.gini_list[i] == ds.best_gini) and (ds.best_split_point == ds.split_points[i]):
            ax[1][0].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.6,
                          label='axvline - full height')
            ax[1][0].plot(ds.split_points[i], ds.gini_list[i], color="black", marker='.', ms=6)
        else:
            ax[1][0].axvline(x=ds.split_points[i], color='black',linestyle="dashed", alpha=0.2 , label='axvline - full height')
            ax[1][0].plot(ds.split_points[i], ds.gini_list[i], color="black", marker='.', ms=5)
    ax[1][0].set_title("Best split for learning data")

    ax[0][1].scatter(new_x_test[:], arr_test, c=y_test, cmap="coolwarm")
    fig.suptitle("Tytul okna")
    ax[0][1].set_title("Test data")

    ax[1][1].scatter(new_x_test[:], arr_test, c=ds_predict, cmap="coolwarm")
    for i in range(len(ds.gini_list)):
        if (ds.gini_list[i] == ds.best_gini) and (ds.best_split_point == ds.split_points[i]):
            ax[1][1].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.6,
                          label='axvline - full height')
            ax[1][1].plot(ds.split_points[i], ds.gini_list[i], color="black", marker='.', ms=6)
        else:
            ax[1][1].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.2 , label='axvline - full height')
            ax[1][1].plot(ds.split_points[i], ds.gini_list[i], color="black", marker='.', ms=5)
    ax[1][1].set_title("Test data split prediction")

    plot.show()

