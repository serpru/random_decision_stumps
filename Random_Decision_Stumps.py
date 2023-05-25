import copy
import random

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from Gini import Gini


class OurRDS(BaseEstimator):

    split_point = 0
    gini = 0
    num_of_splits = 0
    best_gini = 1
    best_error = 1
    best_split_point = 0
    gini_list = []
    split_points = []
    direction = 0
    best_direction = 0
    error_dir_0 = 0
    error_dir_1 = 0
    error_list = []
    error_list_dir0 = []
    error_list_dir1 = []
    prev_split_point = -1
    prev_gini = -1
    has_prev_gini = False

    def __init__(self, *args, num_of_splits, random_state, prev_split_point=None, prev_gini=None):
        self.gini = Gini()
        self.num_of_splits = num_of_splits
        random.seed(a=random_state)
        if (prev_split_point != None) and (prev_gini != None):
            self.prev_split_point = prev_split_point
            self.prev_gini = prev_gini
            self.has_prev_gini = True
        pass

    def fit(self, X, y):
        # 1. Podzielic na podlisty lewa / prawa, policzyc licznosc w kazdej z nich i wyliczyć prawdopodobieństwo
        # 2. Wykonać split
        # 3. Obliczyć gini
        # 4. Powtarzać tyle razy, ile mamy do wyboru punktów splitu
        # 5. Wybrać punkt splitu z najmniejszym gini

        self.split_point = 0
        self.best_gini = 1
        self.best_error = 1
        self.best_split_point = 0
        self.gini_list = []
        self.split_points = []
        self.direction = 0
        self.best_direction = 0
        self.error_dir_0 = 0
        self.error_dir_1 = 0

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

        if self.has_prev_gini:
            split_points_list.pop()
            split_points_list.append(self.prev_split_point)
            #self.best_gini = self.prev_gini
            split_points_list.sort()

        current_split_point = range_list[0]

        counter = 0
        for i in split_points_list:
            print(f"Doing magic on split point index {counter}")
            counter += 1
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

        self.classifiers = []
        for n in range(self.n_classifiers):
            self.classifiers.append(OurRDS(num_of_splits=self.num_of_splits, random_state=self.random_state))
        pass


    def fit(self, X, y):
        #   x_learn[i][j] i-index kazdego X, j-index cechy dla danego X
        #   TODO dla kazdego classifiera przeprowadzic fit
        x_learn = self.pick_random_features(X)

        # print(x_learn[:])
        # print("x_learn pojedyncza kolumna")
        # print(x_learn[0][0])

        x_learn2 = np.zeros(shape=(len(x_learn[0]), len(x_learn)))

        for row in range(len(x_learn)):
            for col in range(len(x_learn[0])):
                x_learn2[col][row] = x_learn[row][col]

        # print(x_learn2[0][0])
        # print("x_learn2")
        # print(x_learn2)
        #
        # print("x_learn2[0] i [1]")
        # print(x_learn2[0])
        # print(x_learn2[1])
        # print("x_learn[0] i [1]")
        # print(x_learn[0])
        # print(x_learn[1])
        for c in range(self.n_classifiers):
            # print(x_learn2[c])
            print(f"Classifier index {c} at work...")
            self.classifiers[c].fit(x_learn2[c], y)

        return self


    def predict(self, X):
        Y = []
        pred_list = []
        for classifier in self.classifiers:
            pred_list.append(classifier.predict(X))


        res = []
        for i in range(len(pred_list[0])):
            res.append(0)

        for i in range(len(pred_list)):
            for j in range(len(pred_list[i])):
                res[j] += pred_list[i][j]

        for i in range(len(res)):
            if res[i] / self.n_classifiers > 0.5:
                Y.append(1)
            else:
                Y.append(0)

        return Y

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

        x_learn = np.zeros(shape=(X.shape[0], len(x_indexes)))

        for i in range(len(x_learn)):
            for x in range(len(x_learn[i])):
                x_learn[i][x] = X[i][x_indexes][x]

        #   x_learn[i][j] i-index kazdego X, j-index cechy dla danego X

        #   To ponizej listuje jeden wiersz. Kazda kolumna to sa te randomowe indexy cech dla kazdego classifiera
        #   TODO zwracac cala taka liste X


        # print("X.shape")
        # print(X.shape)
        # print(x_indexes)
        # print(x_learn.shape)
        # print("x_learn [0] i [1]")
        # print(x_learn[0])
        # print(x_learn[1])
        #
        # print("X[0] i [1]")
        # print(X[0])
        # print(X[1])
        #
        # print("x_learn size")
        # print(len(x_learn))
        # print(len(x_learn[0]))

        return x_learn