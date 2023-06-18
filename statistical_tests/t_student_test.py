from scipy import stats
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from Random_Decision_Stumps import DecisionStump
import csv

from tabulate import tabulate

file = "../results.npy"

headers = ["Gaussian", "DecisionTree", "DecisionStump"]
classifiers = [GaussianNB(), DecisionTreeClassifier(), DecisionStump(random_state=1111, num_of_splits=20)]

res = np.load(file)
res_one = res[0]

print(res_one.shape)
print(res.shape)

print(res)

alpha = 0.05

t_stat = np.zeros((len(classifiers), len(classifiers)))
p_val = np.zeros((len(classifiers), len(classifiers)))
better = np.zeros((len(classifiers), len(classifiers))).astype(bool)
better_score = np.zeros((len(classifiers), len(classifiers)))
better_score = []
mean_scores = []
std_scores = []
for k in range(len(res)):
    better_score.append([])
    mean_scores.append([])
    std_scores.append([])

better_score2 = []
for k in range(res.shape[0]):
    better_score2.append([])
    for i in range(len(classifiers)):
        better_score2[k].append([])

counter = 0
for k in range(len(res)):
    res_one = res[k]
    for i in range(len(classifiers)):
        mean_scores[k].append(round(np.mean(res_one[:,i]),3))
        std_scores[k].append(round(np.std(res_one[:, i]), 3))
        for j in range(len(classifiers)):
            _t_stat, _p_val = stats.ttest_rel(res_one[:, i], res_one[:, j])
            t_stat[i,j] = _t_stat
            p_val[i,j] = _p_val
            better[i,j] = np.mean(res_one[:,i]) > np.mean(res_one[:,j])
            if better[i,j]:
                better_score[k].append(f"{headers[i]} better than {headers[j]} with {round(np.mean(res_one[:,i]),3)}")
                better_score2[k][i].append(j)


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
        print(f"{headers[m]}: {mean_scores[k][m]} ({std_scores[k][m]})")
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

row = ["", headers[0], headers[1], headers[2]]
print(f"         |  {headers[0]}     | {headers[1]}  | {headers[2]}")
for k in range(res.shape[0]):
    print(f"dataset{k} | {mean_scores[k][0]} ({std_scores[k][0]}) | {mean_scores[k][1]} ({std_scores[k][1]}) | {mean_scores[k][2]} ({std_scores[k][2]})")
    print(f"          |     {better_score2[k][0]}     |      {better_score2[k][1]}      |        {better_score2[k][2]}")


f = open('test.csv', 'w')
writer = csv.writer(f)


writer.writerow(row)

for i in range(res.shape[0]):
    row = [f"dataset {i}", f"{mean_scores[i][0]} ({std_scores[i][0]})", f"{mean_scores[i][1]} ({std_scores[i][1]})",
           f"{mean_scores[i][2]} ({std_scores[i][2]})"]
    writer.writerow(row)
    better_row = []
    better_row.append("")
    for x in range(len(classifiers)):
        if len(better_score2[i][x]) >= 2:
            better_row.append("all")
        elif len(better_score2[i][x]) < 2 and len(better_score2[i][x]) > 0:
            better_row.append(f"{better_score2[i][x][0]}")
        else:
            better_row.append("none")
    writer.writerow(better_row)


f.close()