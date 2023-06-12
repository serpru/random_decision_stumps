import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class RDSEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classifiers=5, num_of_splits=20, random_state=1):
        self.n_classifiers = n_classifiers
        self.num_of_splits = num_of_splits
        self.random_state = random_state
        self.x_indexes = []

        np.random.seed(self.random_state)

        self.classifiers = []
        for n in range(self.n_classifiers):
            self.classifiers.append(DecisionStump(num_of_splits=self.num_of_splits, random_state=self.random_state))
        pass


    def fit(self, X, y):
        x_learn = self.pick_random_features(X)
        # print(self.x_indexes)

        for c in range(self.n_classifiers):
            self.classifiers[c].fit(x_learn[:, c], y)

        # dopasowanie -- obliczenie wag
        acc = []
        for c_id, classifier in enumerate(self.classifiers):
            this_X = X[:,self.x_indexes[c_id]].reshape(-1,1)
            p = classifier.predict(this_X)
            acc.append(accuracy_score(y,p))
        # print(acc)
        self.weights = acc / np.sum(acc)
        # print(np.sum(acc))
        # print(self.weights)
        # print("weights len")
        # print(len(self.weights))
        return self


    def predict(self, X):
        pred_list = []
        for c, classifier in enumerate(self.classifiers):
            #   Reshape (-1, 1) changes X from [1, 2, 3, ...] to [[1], [2], [3], ...]
            #   Most sklearn classifiers use this data format
            this_X = X[:, self.x_indexes[c]].reshape(-1, 1)
            pred_list.append(classifier.predict(this_X))

        pred_list = np.array(pred_list)

        #   Changing from 0 : 1 to -1 : 1 to prepare for adding weights
        pred_list[pred_list == 0] = -1

        #   Adding weights to samples
        pred_list = pred_list * self.weights[:, None] # None?

        # print(pred_list)
        sum_pred = np.sum(pred_list, axis=0)
        # print(sum_pred)
        pred = (sum_pred >= 0).astype(int)
        # print(pred)


        # avg_pred = np.mean(pred_list, axis=0)
        # pred = (avg_pred >= 0.5).astype(int)
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

        self.x_indexes = x_indexes
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
        #   Split point splits data, then gini impurity is calculated for each part (above/below split) in search
        #   for split point with lowest gini impurity
        #   Classification error is also calculated to determine which way to classify samples (0 | 1 or 1 | 0)
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
        #   X is an array of one feature
        #   Depending on best_direction value, y is set with 0 or 1 above the best_split_point
        if self.best_direction == 1:
            y = np.ones(shape=X.shape[0], dtype=int)
        else:
            y = np.zeros(shape=X.shape[0], dtype=int)
        for x_id, x in enumerate(X):
            if x >= self.best_split_point:
                y[x_id] = 1 - self.best_direction
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
