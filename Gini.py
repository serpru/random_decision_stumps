import numpy as np


class Gini:

    y_counts = 0
    org = -1
    gini_weighted = -1
    gini_gain = -1
    classification_error = 0
    pred_correct = 0
    pred_wrong = 0

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

    def gini_temp(self, left_y_split, right_y_split, left_y_org, right_y_org, direction):
        g_res = 0
        arr_y_org = [left_y_org, right_y_org]
        L = len(arr_y_org[0])+len(arr_y_org[1])

        leaves_split = [left_y_split, right_y_split]

        # for i in range(len(arr_y)):
        #     counts = np.unique(arr_y[i], return_counts=1)
        #     print("Counts")
        #     print(counts)
        #     if len(counts[0]) < 2:
        #         continue
        #
        #     prob_zero, prob_one = self.prob(arr_y_org[i])
        #     print("Probs")
        #     print(prob_zero)
        #     print(prob_one)
        #
        #     g = (len(arr_y[i]) / len(y_org)) * (1 - (prob_zero * prob_zero) - (prob_one * prob_one))
        #     print(g)
        #     g_res += g

          #Sprawdzenie czy predykcja jest poprawna
        for i in range(len(leaves_split)):
            n = len(leaves_split[i])
            if n == 0:
                continue

            prediction_correct = 0
            prediction_wrong = 0
            for j in range(len(leaves_split[i])):
                #   For now arr_y isn't the array after split. I think it needs to be after split so it can
                #   evaluate like so: is value after split == value before? and is value before right with direction?
                prediction_correct += ((leaves_split[i][j] == arr_y_org[i][j]) and (arr_y_org[i][j] == direction))
                prediction_wrong += ((leaves_split[i][j] != arr_y_org[i][j]) and (arr_y_org[i][j] == direction))

            self.pred_correct = prediction_correct
            self.pred_wrong = prediction_wrong

            prob_correct = prediction_correct/n
            prob_wrong = prediction_wrong/n

            gini = 1 - prob_correct*prob_correct - prob_wrong*prob_wrong
            g_res += (len(leaves_split[i])/L) * gini

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

    def class_error(self, wrong_len, all_len):
        return wrong_len/all_len

