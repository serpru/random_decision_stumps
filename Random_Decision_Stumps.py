import copy
import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from Gini import Gini


class OurRDS(BaseEstimator):
    def __init__(self, *args, num_of_splits, random_state):
        self.gini = Gini()
        self.num_of_splits = num_of_splits
        random.seed(a=random_state)

        self.split_point = 0
        self.num_of_splits = num_of_splits
        self.best_gini = 1
        self.best_error = 1
        self.best_split_point = 0
        self.gini_list = []
        self.split_points = []
        self.direction = 0
        self.best_direction = 0
        self.error_dir_0 = 0
        self.error_dir_1 = 0
        self.error_list = []
        self.error_list_dir0 = []
        self.error_list_dir1 = []



    def fit(self, X, y):
        # 1. Podzielic na podlisty lewa / prawa, policzyc licznosc w kazdej z nich i wyliczyć prawdopodobieństwo
        # 2. Wykonać split
        # 3. Obliczyć gini
        # 4. Powtarzać tyle razy, ile mamy do wyboru punktów splitu
        # 5. Wybrać punkt splitu z najmniejszym gini

        range_list = []
        for i in range(len(X)):
            range_list.append(X[i])

        # for i in X:
        #     range_list.append(i[0])
        range_list.sort()
        mean = (abs(range_list[0]) + abs(range_list[-1])) / self.num_of_splits

        split_points_list = []
        for i in range(self.num_of_splits):
            split_points_list.append(range_list[0] + i*mean)

        current_split_point = range_list[0]

        for i in split_points_list:
            self.split_point = i
            current_split_point = i

            left_leaf, right_leaf = self.slice(X, y, self.split_point)

            #   Direction 0
            classified_left_leaf_d0, classified_right_leaf_d0 = self.split(left_leaf[1], right_leaf[1])

            current_gini_dir0 = self.gini.calculate_gini(classified_left_leaf_d0, classified_right_leaf_d0, left_leaf[1], right_leaf[1], direction=0)

            self.error_dir_0 = self.gini.class_error(self.gini.pred_wrong, len(y))

            #   Direction 1
            classified_left_leaf_d1, classified_right_leaf_d1 = self.split(left_leaf[1], right_leaf[1], direction=1)
            current_gini_dir1 = self.gini.calculate_gini(classified_left_leaf_d1, classified_right_leaf_d1, left_leaf[1], right_leaf[1],
                                               direction=1)

            self.error_dir_1 = self.gini.class_error(self.gini.pred_wrong, len(y))

            class_error = 1
            #   Choosing best gini and best direction
            if self.best_gini > current_gini_dir1:
                self.best_gini = current_gini_dir1
                self.best_split_point = current_split_point
                if self.error_dir_0 > self.error_dir_1:
                    self.direction = 1
                    self.best_error = self.error_dir_1
                else:
                    self.direction = 0
                    self.best_error = self.error_dir_0

            final_gini = current_gini_dir0
            if current_gini_dir0 > current_gini_dir1:
                final_gini = current_gini_dir1

            #   Adding gini and split point to the list
            self.split_points.append(current_split_point)
            self.gini_list.append(final_gini)

            self.error_list_dir0.append(self.error_dir_0)
            self.error_list_dir1.append(self.error_dir_1)

            #current_split_point += mean

        return self

    def predict(self, X):
        arr = []
        print(self.best_split_point)
        for i in range(len(X)):
            if X[i][0] >= self.best_split_point:
                arr.append(1 - self.direction)
            else:
                arr.append(0 + self.direction)
        return np.array(arr)

    def slice(self, X, y, a):
        #   Slices data into two subsets at splitting point "a"
        left_x = []
        right_x = []
        left_y = []
        right_y = []

        for i in range(len(X)):
            if X[i] >= a:
                right_x.append(X[i])
                right_y.append(y[i])
            else:
                left_x.append(X[i])
                left_y.append(y[i])
        return [left_x, left_y], [right_x, right_y]

    def split(self, left_y, right_y, direction=0):
        # Splits label array into two sub-arrays
        # One has assigned 0 to it, other 1, based on direction
        new_left_y = []
        new_right_y = []
        for i in left_y:
            new_left_y.append(0+direction)
        for i in right_y:
            new_right_y.append(1-direction)
        return new_left_y, new_right_y


class RDSEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classifiers=5, num_of_splits=20, random_state=1):
        self.n_classifiers = n_classifiers
        self.num_of_splits = num_of_splits
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.classifiers = []
        for n in range(self.n_classifiers):
            self.classifiers.append(DecisionStump(num_of_splits=self.num_of_splits, random_state=self.random_state))
        pass


    def fit(self, X, y):
        x_learn = self.pick_random_features(X)

        for c in range(self.n_classifiers):
            self.classifiers[c].fit(x_learn[:, c], y)

        return self


    def predict(self, X):
        Y = []
        pred_list = []
        for classifier in self.classifiers:
            pred_list.append(classifier.predict(X))

        pred_list = np.array(pred_list)

        avg_pred = np.mean(pred_list, axis=0)
        pred = (avg_pred >= 0.5).astype(int)
        return pred

    def pick_random_features(self, X):
        #   Picks random n_classifiers number of features from X
        #   For example if X.shape = (100, 20), n_classifiers number of features is chosen for each X
        #   It results in x_learn.shape = (100, n_classifiers) sized array
        np.random.seed(self.random_state)
        is_replace = False
        if X.shape[1] < self.n_classifiers:
            is_replace = True

        x_indexes = np.random.choice(
            a=int(X.shape[1]),
            size=self.n_classifiers,
            replace=is_replace
        )

        x_learn = X[:, x_indexes]
        return x_learn

class DecisionStump(BaseEstimator):
    def __init__(self, *args, num_of_splits, random_state):
        self.num_of_splits = num_of_splits
        self.best_gini = 1
        self.best_error = 1
        self.best_split_point = 0
        self.gini_list = []
        self.split_points = []
        self.best_direction = 0
        self.error_dir_0 = 1
        self.error_dir_1 = 1
        self.error_list_dir0 = []
        self.error_list_dir1 = []

    def fit(self, X, y):

        #   Choosing split points. Split points are put across the dataset with equal distance between each other
        range_list = []
        for i in range(len(X)):
            range_list.append(X[i])

        range_list.sort()

        mean = (abs(range_list[0]) + abs(range_list[-1])) / self.num_of_splits

        split_points_list = []
        for i in range(self.num_of_splits):
            split_points_list.append(range_list[0] + i*mean)

        for point in split_points_list:
            current_split_point = point

            indexes_above_split = []
            indexes_below_split = []
            for i in range(len(X)):
                if X[i] >= current_split_point:
                    indexes_above_split.append(i)
                else:
                    indexes_below_split.append(i)

            #   Ignore iteration if there is no data below or above split
            if not indexes_above_split or not indexes_below_split:
                continue
            else:
                counts_above = np.unique(ar=y[indexes_above_split], return_counts=True)
                counts_below = np.unique(ar=y[indexes_below_split], return_counts=True)

                #   Failsafe when counts_above[1] has one-element list
                if len(counts_above[1]) < 2:
                    gini_above = self.Gini(counts_above[1], 0)
                else:
                    gini_above = self.Gini(counts_above[1][0], counts_above[1][1])

                #   Same for counts_below[1]
                if len(counts_below[1]) < 2:
                    gini_below = self.Gini(counts_below[1], 0)
                else:
                    gini_below = self.Gini(counts_below[1][0], counts_below[1][1])

                sum_above = np.sum(counts_above[1])
                sum_below = np.sum(counts_below[1])
                weight_above = sum_above / len(y)
                weight_below = sum_below / len(y)

                wght_gini = self.Wght_Gini(gini_above, gini_below, weight_above, weight_below)

                #   Classify data above and below split point
                #   Direction 0-1 (0 below split, 1 above split)
                Y = np.zeros(shape=X.shape[0], dtype=int)

                Y[indexes_above_split] = 1
                Y[indexes_below_split] = 0

                error_dir_01 = self.Class_Error(y, Y)

                #   Check class error with direction 1-0
                Y[indexes_above_split] = 0
                Y[indexes_below_split] = 1

                error_dir_10 = self.Class_Error(y, Y)

                if wght_gini < self.best_gini:
                    self.best_gini = wght_gini
                    self.best_split_point = current_split_point
                    if error_dir_01 < error_dir_10:
                        self.best_direction = 0
                        if error_dir_01 < self.best_error:
                            self.best_error = error_dir_01
                    else:
                        self.best_direction = 1
                        if error_dir_10 < self.best_error:
                            self.best_error = error_dir_10

                #   Adding gini and split point to the list
                self.split_points.append(current_split_point)
                self.gini_list.append(wght_gini)

                self.error_list_dir0.append(error_dir_01)
                self.error_list_dir1.append(error_dir_10)

        return self

    def predict(self, X):
        if self.best_direction == 1:
            y = np.ones(shape=X.shape[0], dtype=int)
        else:
            y = np.zeros(shape=X.shape[0], dtype=int)
        for i in range(len(X)):
            if X[i][0] >= self.best_split_point:
                y[i] = 1 - self.best_direction
        return y

    def Gini(self, P1, P2):
        # P1 and P2 are the counts for each class after the split
        sum = P1 + P2
        Gini = 2 * (P1 / sum) * (P2 / sum)
        return (Gini)

    def Wght_Gini(self, gini1, gini2, weight1, weight2):
        w_gini = (weight1 * gini1) + (weight2 * gini2)
        return w_gini

    def Class_Error(self, y_org, y):
        wrong = 0
        for i in range(len(y_org)):
            if y_org[i] != y[i]:
                wrong += 1
        return wrong / len(y)
