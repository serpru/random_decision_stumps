from scipy import stats
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from Random_Decision_Stumps import OurRDS, DecisionStump

from tabulate import tabulate

file = "results.npy"

headers = ["Gaussian", "DecisionTree", "DecisionStump"]
classifiers = [GaussianNB(), DecisionTreeClassifier(), DecisionStump(random_state=1111, num_of_splits=20)]

res = np.load(file)
res_one = res[0]

print(res_one.shape)
print(res.shape)

alpha = 0.05

t_stat = np.zeros((len(classifiers), len(classifiers)))
p_val = np.zeros((len(classifiers), len(classifiers)))
better = np.zeros((len(classifiers), len(classifiers))).astype(bool)
better_score = np.zeros((len(classifiers), len(classifiers)))
better_score = []
mean_scores = []
for k in range(len(res)):
    better_score.append([])
    mean_scores.append([])

counter = 0
for k in range(len(res)):
    res_one = res[k]
    for i in range(len(classifiers)):
        mean_scores[k].append(round(np.mean(res_one[:,i]),3))
        for j in range(len(classifiers)):
            _t_stat, _p_val = stats.ttest_rel(res_one[:, i], res_one[:, j])
            t_stat[i,j] = _t_stat
            p_val[i,j] = _p_val
            better[i,j] = np.mean(res_one[:,i]) > np.mean(res_one[:,j])
            if better[i,j]:
                better_score[k].append(f"{headers[i]} better than {headers[j]} with {round(np.mean(res_one[:,i]),3)}")

    significant = p_val < alpha
    significantly_better = significant * better

    # print("\nt-statystyka")
    # print(t_stat)
    # print("\np-val")
    # print(p_val)
    # print("\nMacierz lepszych wyników")
    # print(better)
    # print("\nMacierz istotności statystycznej")
    # print(significantly_better)


    names_column = np.expand_dims(np.array(headers), axis=1)
    t_statistic_table = np.concatenate((names_column, t_stat), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_val), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    better_table = tabulate(np.concatenate(
        (names_column, better), axis=1), headers)
    significantly_better_table = tabulate(np.concatenate(
        (names_column, significantly_better), axis=1), headers)

    print(f"\n#### Test {counter} ####\n")
    print("Wartości średnie wyników:")
    for m in range(len(mean_scores[k])):
        print(f"{headers[m]}: {mean_scores[k][m]}")
    print("\nt-statystyka")
    print(t_statistic_table)
    print("\np-val")
    print(p_value_table)
    print("\nMacierz lepszych wyników")
    print(better_table)
    print("\nMacierz istotności statystycznej")
    print(significantly_better_table)
    print("\nInterpretacja wyników:")
    for text in better_score[k]:
        print(text)



    counter += 1
