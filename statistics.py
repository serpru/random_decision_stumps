from scipy import stats
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from Random_Decision_Stumps import OurRDS

file = "results.npy"

classifiers = [GaussianNB(), DecisionTreeClassifier(), OurRDS(random_state=1234, num_of_splits=20)]

res = np.load(file)
res_one = res[0]
print(res)

print(res_one.shape)

alpha = 0.05

t_stat = np.zeros((len(classifiers), len(classifiers)))
p_val = np.zeros((len(classifiers), len(classifiers)))
better = np.zeros((len(classifiers), len(classifiers))).astype(bool)

for k in range(len(res)):
    res_one = res[k]
    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            _t_stat, _p_val = stats.ttest_rel(res_one[:, i], res_one[:, j])
            t_stat[i,j] = _t_stat
            p_val[i,j] = _p_val
            better[i,j] = np.mean(res_one[:,i]) > np.mean(res_one[:,j])


    significant = p_val < alpha
    significantly_better = significant * better

    print("\nt-statystyka")
    print(t_stat)
    print("\np-val")
    print(p_val)
    print("\nMacierz lepszych wyników")
    print(better)
    print("\nMacierz istotności statystycznej")
    print(significantly_better)
