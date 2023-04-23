import numpy as np


class Gini:

    y_counts = 0
    org = -1
    gini_weighted = -1
    gini_gain = -1

    def __init__(self):
        # TODO
        pass

    # Unbalanced something \/
    def original_gini(self, y):
        self.y_counts = np.unique(y, return_counts=1)
        zeros_percent = self.y_counts[1][0] / (self.y_counts[1][0] + self.y_counts[1][1])
        ones_percent = 1 - zeros_percent
        self.org = 1 - pow(zeros_percent, 2) - pow(ones_percent, 2)
        return self

    def gini_weighted(self, left, right):
        # TODO Fix this, it takes wrong probability values
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

    def gini_temp(self, arr_y, y_org):
        g_res = 0

        for y in arr_y:
            counts = np.unique(y, return_counts=1)
            print("Counts")
            print(counts)
            if len(counts[0]) < 2:
                continue

            prob_zero, prob_one = self.prob(y)
            print("Probs")
            print(prob_zero)
            print(prob_one)

            g = (len(y) / len(y_org)) * (1 - (prob_zero * prob_zero) - (prob_one * prob_one))
            print(g)
            g_res += g
        return g_res

    def calc_gini_gain(self):
        self.gini_gain = self.org - self.gini_weighted

    def prob(self, y):
        unique_arr = np.unique(y, return_counts=1)

        zeros = unique_arr[1][0]
        ones = unique_arr[1][1]

        prob_zero = zeros / len(y)
        prob_one = ones / len(y)

        return prob_zero, prob_one
