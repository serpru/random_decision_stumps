import numpy as np
import matplotlib.pyplot as plot


res = np.load(file="../experiment_results/n_classifiers_results.npy")
num_of_splits = np.load(file="../experiment_results/n_classifiers.npy")

print(res.shape)
print(num_of_splits)


mean_scores = np.zeros((res.shape[0], res.shape[1], 1))

print(mean_scores.shape)

mean = np.zeros(shape=(10))

for i in range(len(res)):
    for j in range(len(res[1])):
        mean_scores[i][j] = np.mean(res[i][j])

for i in range(mean_scores.shape[0]):
    mean[i] = np.mean(mean_scores[i])

print(mean)
print(mean.shape)

#   Plotting single experiment
fig, ax = plot.subplots(
    nrows=1,
    ncols=1,
    sharex=True,
    sharey=True,
)

ax.scatter(num_of_splits, mean, c="black")
fig.suptitle(f"Experiment - number of classifiers per ensemble.")
ax.set_title(f"Mean accuracy score for each ensemble")

plot.show()