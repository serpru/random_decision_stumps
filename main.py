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

    # Random Seed
    rnd_seed = 12442
    # 12442
    # 9865

    #   Number of splits to check per Stump
    num_of_splits = 20

    # Dataset
    x, y = datasets.make_classification(
        weights=None,
        n_samples=400,
        n_features=1,
        n_classes=2,
        n_informative=1,
        n_redundant=0,
        n_repeated=0,
        flip_y=0.3,
        random_state=rnd_seed,
        n_clusters_per_class=1
    )


    #   Uncomment this to quickly invert y's 0 1 values
    #   to test if the Stump correctly assigns classification direction
    # inverted_y = []
    #
    # for item in y:
    #     if item == 0:
    #         inverted_y.append(1)
    #     else:
    #         inverted_y.append(0)
    #
    # y = inverted_y

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
        n_repeats=10,
        random_state=rnd_seed,
    )

    # Single experiment
    ds = OurRDS(num_of_splits=num_of_splits, random_state=rnd_seed)
    ds.fit(x_learn, y_learn)

    ds_learn_y = ds.predict(x_learn)
    ds_predict = ds.predict(x_test)
    a_score = accuracy_score(y_test, ds_predict)


    #   Repeated Stratified K Fold
    a_score_ords = []
    for i, (train_index, test_index) in enumerate(rskf.split(x, y)):
        ords = OurRDS(num_of_splits=20, random_state=rnd_seed)
        ords.fit(x[train_index], y[train_index])
        predict_ords = ords.predict(x[test_index])
        a_score_ords.append(accuracy_score(y[test_index], predict_ords))

    print("\n#### Fold results ####")
    print(a_score_ords)

    mean_score = np.mean(a_score_ords)
    std_score = np.std(a_score_ords)

    print("Mean accuracy score: " "%.3f" % mean_score)
    print("Standard deviation: " "%.3f" % std_score)

    print("\n#### Single experiment results ####")
    print("Best Gini: " "%.3f" % ds.best_gini)
    print("Split point at best gini: " "%.3f" %ds.best_split_point)
    print("Classification error at best split point: " "%.3f" % ds.best_error)
    print("Accuracy score: " "%.3f" % a_score)


    #   Plotting single experiment
    fig, ax = plot.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        sharey=True,
    )

    #   Random Y axis placement for data. For better data readability on the plot, only.
    random.seed(rnd_seed)
    arr_learn = []
    for x in x_learn:
        n = random.triangular(-0.4, 0.7)
        arr_learn.append(0.7 + n)

    arr_test = []
    for x in x_test:
        n = random.triangular(0.1, 0.9)
        arr_test.append(0.5 + n)

    ax[0][0].scatter(x_learn[:, 0], arr_learn, c=y_learn, cmap="coolwarm")
    fig.suptitle(f"Accuracy score of test split: " "%.3f" % a_score)
    ax[0][0].set_title("Learning data")

    ax[1][0].scatter(x_learn[:, 0], arr_learn, c=ds_learn_y, cmap="coolwarm")
    for i in range(len(ds.gini_list)):
        if ds.best_split_point == ds.split_points[i]:
            ax[1][0].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')
        else:
            ax[1][0].axvline(x=ds.split_points[i], color='black',linestyle="dashed", alpha=0.2 , label='axvline - full height')
        gini_marker = ax[1][0].plot(ds.split_points[i], ds.gini_list[i], color="black", marker='.', ms=5)
        error_dir0_marker = ax[1][0].plot(ds.split_points[i], ds.error_list_dir0[i], color="blue", marker='.', ms=3)
        error_dir1_marker = ax[1][0].plot(ds.split_points[i], ds.error_list_dir1[i], color="red", marker='.', ms=3)
    ax[1][0].set_title("Best split for learning data")

    ax[0][1].scatter(x_test[:, 0], arr_test, c=y_test, cmap="coolwarm")

    ax[0][1].set_title("Test data")

    ax[1][1].scatter(x_test[:, 0], arr_test, c=ds_predict, cmap="coolwarm")
    for i in range(len(ds.gini_list)):
        if (ds.gini_list[i] == ds.best_gini) and (ds.best_split_point == ds.split_points[i]):
            ax[1][1].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.7,
                          label='axvline - full height')
    ax[1][1].set_title("Test data split prediction")

    plot.show()

