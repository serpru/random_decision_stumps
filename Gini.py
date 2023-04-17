import numpy as np


class Gini:

    y_counts = 0
    org = -1
    gini_weighted = -1
    gini_gain = -1

    def __init__(self):
        # TODO
        pass

    def original_gini(self, y):
        self.y_counts = np.unique(y, return_counts=1)
        zeros_percent =  self.y_counts[1][0] / (self.y_counts[1][0] + self.y_counts[1][1])
        ones_percent = 1 - zeros_percent
        self.org = 1 - pow(zeros_percent, 2) - pow(ones_percent, 2)
        return self

    def gini_weighted(self, left, right):
        left_counts = np.unique(left, return_counts=1)
        zeros_percent = left_counts[1][0] / (left_counts[1][0] + left_counts[1][1])
        ones_percent = 1 - zeros_percent
        gini_left = 1 - pow(zeros_percent, 2) - pow(ones_percent, 2)

        right_counts = np.unique(right, return_counts=1)
        zeros_percent = right_counts[1][0] / (right_counts[1][0] + right_counts[1][1])
        ones_percent = 1 - zeros_percent
        gini_right = 1 - pow(zeros_percent, 2) - pow(ones_percent, 2)

        self.gini_weighted = (gini_left + gini_right) / 2
        return self

    def calc_gini_gain(self):
        self.gini_gain = self.org - self.gini_weighted

