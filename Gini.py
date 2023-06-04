import numpy as np


class Gini:

    def __init__(self):
        self.classification_error = 0
        self.pred_correct = 0
        self.pred_wrong = 0

    def calculate_gini(self, left_y_split, right_y_split, left_y_org, right_y_org, direction):
        g_res = 0
        arr_y_org = [left_y_org, right_y_org]
        L = len(arr_y_org[0])+len(arr_y_org[1])

        leaves_split = [left_y_split, right_y_split]

        #   Checking how correct was prediction
        for i in range(len(leaves_split)):
            n = len(leaves_split[i])
            if n == 0:
                continue

            prediction_correct = 0
            prediction_wrong = 0
            for j in range(len(leaves_split[i])):
                prediction_correct += ((leaves_split[i][j] == arr_y_org[i][j]) and (arr_y_org[i][j] == direction))
                prediction_wrong += ((leaves_split[i][j] != arr_y_org[i][j]) and (arr_y_org[i][j] == direction))

            self.pred_correct = prediction_correct
            self.pred_wrong = prediction_wrong

            prob_correct = prediction_correct/n
            prob_wrong = prediction_wrong/n

            gini = 1 - prob_correct*prob_correct - prob_wrong*prob_wrong
            g_res += (len(leaves_split[i])/L) * gini

        return g_res

    def prob(self, y):
        unique_arr = np.unique(y, return_counts=1)

        zeros = unique_arr[1][0]
        ones = unique_arr[1][1]

        prob_zero = zeros / len(y)
        prob_one = ones / len(y)

        return prob_zero, prob_one

    def class_error(self, wrong_len, all_len):
        return wrong_len/all_len

