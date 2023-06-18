import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Random_Decision_Stumps import DecisionStump
import matplotlib.pyplot as plot

rnd_seed = 1111
num_of_splits = 15
feature_id = 18

data_list = os.listdir(path="../datasets20f")

file_path = "../datasets20f/" + data_list[0]
data = np.loadtxt(file_path, delimiter=",", dtype=str)

X = []
for item in data:
    X.append(item[:-1])

X = np.array(X)
X = X.astype(dtype=float)

y = []
for item in data:
    y.append(item[-1])
y = np.array(y)
y = y.astype(dtype=int)

# Split train/test
x_learn, x_test, y_learn, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    train_size=0.8,
    random_state=rnd_seed,
)

ds = DecisionStump(num_of_splits=num_of_splits, random_state=rnd_seed)
ds.fit(x_learn[:, feature_id], y_learn)
pred = ds.predict(x_test[:, feature_id])

learned_y = ds.predict(x_learn[:, feature_id])

print(x_learn[:, feature_id])

print("x_learn size")
print(len(x_learn))

print("x_test")
print(x_test)
print("predict")
print(pred)
print(y_test)

print("Best direction")
print(ds.best_direction)

print("Best gini")
print(ds.best_gini)

print("Best error")
print(ds.best_error)

print("Best split point")
print(ds.best_split_point)

a_score = accuracy_score(y_test, pred)
print(a_score)

fig, ax = plot.subplots(
    nrows=2,
    ncols=2,
    sharex=True,
    sharey=True,
)

ax[0][0].scatter(x_learn[:, feature_id], x_learn[:, 1], c=y_learn, cmap="coolwarm")
fig.suptitle(f"Single Decision Stump\nAccuracy score of test split: {a_score}\n feature index {feature_id}")
ax[0][0].set_title("Learning data")

ax[1][0].scatter(x_learn[:, feature_id], x_learn[:, 1], c=learned_y, cmap="coolwarm")
for i in range(len(ds.gini_list)):
    if ds.best_split_point == ds.split_points[i]:
        ax[1][0].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.7,
                      label='axvline - full height')
    else:
        ax[1][0].axvline(x=ds.split_points[i], color='black',linestyle="dashed", alpha=0.2 , label='axvline - full height')
    gini_marker = ax[1][0].plot(ds.split_points[i], ds.gini_list[i], color="black", marker='.', ms=5)
    error_dir0_marker = ax[1][0].plot(ds.split_points[i], ds.error_list_dir0[i], color="blue", marker='.', ms=3, alpha=0.5)
    error_dir1_marker = ax[1][0].plot(ds.split_points[i], ds.error_list_dir1[i], color="red", marker='.', ms=3, alpha=0.5)
ax[1][0].set_title("Best split for learning data")

ax[0][1].scatter(x_test[:, feature_id], x_test[:, 1], c=y_test, cmap="coolwarm")

ax[0][1].set_title("Test data")

ax[1][1].scatter(x_test[:, feature_id], x_test[:, 1], c=pred, cmap="coolwarm")
for i in range(len(ds.gini_list)):
    if (ds.gini_list[i] == ds.best_gini) and (ds.best_split_point == ds.split_points[i]):
        ax[1][1].axvline(x=ds.split_points[i], color='black', linestyle="dashed", alpha=0.7,
                      label='axvline - full height')
ax[1][1].set_title("Test data split prediction")

plot.show()
