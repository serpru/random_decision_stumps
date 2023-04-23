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
    best_split_point = 0
    gini_list = []
    gini_list_other_direction = []
    split_points = []
    direction = 0

    def __init__(self, split_point, num_of_splits, random_state):
        self.split_point = split_point
        self.gini = Gini()
        self.num_of_splits = num_of_splits
        random.seed(a=random_state)

        pass

    def fit(self, X, y):
        # TODO
        # 1. Podzielic na podlisty lewa / prawa, policzyc licznosc w kazdej z nich i wyliczyć prawdopodobieństwo
        # 2. Wykonać split
        # 3. Obliczyć gini
        # 4. Powtarzać tyle razy, ile mamy do wyboru punktów splitu
        # 5. Wybrać punkt splitu z najmniejszym gini

        new_y = []

        range_list = []
        for i in X:
            range_list.append(i[0])
        range_list.sort()
        mean = (abs(range_list[0]) + abs(range_list[-1])) / self.num_of_splits

        current_split_point = range_list[0]
        #random.triangular(range_list[0], range_list[-1])
        for i in range(self.num_of_splits):
            current_split_point += mean
            self.split_points.append(current_split_point)
            self.split_point = current_split_point

            left_data, right_data = self.pre_split(X, y, self.split_point)

            #   Direction 0
            new_left_y, new_right_y = self.split(left_data[1], right_data[1])

            current_gini = self.gini.gini_temp([left_data[1], right_data[1]], y)

            if self.best_gini > current_gini:
                self.best_gini = current_gini
                self.best_split_point = current_split_point

            self.gini_list.append(current_gini)



        return self

    def predict(self, X):
        arr = []
        for i in range(len(X)):
            if X[i] >= self.best_split_point:
                arr.append(1)
            else:
                arr.append(0)
        return arr

    def pre_split(self, X, y, a):
        #   Splits data into two subsets at splitting point "a"
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


    def prob(self, y):
        unique_arr = np.unique(y, return_counts=1)

        zeros = unique_arr[1][0]
        ones = unique_arr[1][1]

        prob_zero = zeros / len(y)
        prob_one = ones / len(y)

        return prob_zero, prob_one
