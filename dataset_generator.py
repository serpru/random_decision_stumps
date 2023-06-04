import numpy as np
from sklearn import datasets



# Random Seed
rnd_seed = [12442, 2222, 6245, 8452, 5211, 8923, 6821, 1111, 2895, 8712, 9999, 5875, 3863, 8572, 5729, 4721, 3156, 3333, 8301, 2001]
# 12442
# 9865

for i in range(len(rnd_seed)):
    # Dataset
    x, y = datasets.make_classification(
        n_samples=400,
        n_features=20,
        n_classes=2,
        n_informative=20,
        n_redundant=0,
        n_repeated=0,
        random_state=rnd_seed[i]
    )

    dataset = np.concatenate((x, y[:, np.newaxis]), axis=1)

    np.savetxt(
        f"./datasets20f/dataset{i}.csv",
        dataset,
        delimiter=",",
        fmt=["%.5f" for i in range(x.shape[1])] + ["%i"],
    )
