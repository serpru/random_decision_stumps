import copy
import random

import numpy as np
from sklearn.base import BaseEstimator
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

    def __init__(self, num_of_splits, random_state):
        self.gini = Gini()
        self.num_of_splits = num_of_splits
        random.seed(a=random_state)
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
        for i in X:
            range_list.append(i[0])
        range_list.sort()
        mean = (abs(range_list[0]) + abs(range_list[-1])) / self.num_of_splits

        current_split_point = range_list[0]
        for i in range(self.num_of_splits):
            self.split_point = current_split_point

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

            current_split_point += mean

        return self

    def predict(self, X):
        arr = []
        for i in range(len(X)):
            if X[i][0] >= self.best_split_point:
                arr.append(1 - self.direction)
            else:
                arr.append(0 + self.direction)
        return arr

    def slice(self, X, y, a):
        #   Slices data into two subsets at splitting point "a"
        left_x = []
        right_x = []
        left_y = []
        right_y = []

        for i in range(len(X)):
            if X[i][0] >= a:
                right_x.append(X[i][0])
                right_y.append(y[i])
            else:
                left_x.append(X[i][0])
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
