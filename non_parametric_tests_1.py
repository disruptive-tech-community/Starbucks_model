import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = './data/'
# Bootstrapping

# Bootstrapping is used to estimate sampling distributions by using the
# actually collected data to generate new samples that could have been
# hypothetically collected. In a standard bootstrap, a bootstrapped sample
# means drawing points from the original data with replacement until we get as
# many points as there were in the original data. Essentially, we're treating
# the original data as the population: without making assumptions about the
# original population distribution, using the original data as a model of the
# population is the best that we can do.

# Taking a lot of bootstrapped samples allows us to estimate the sampling
# distribution for various statistics on our original data. For example,
# let's say that we wanted to create a 95% confidence interval for the 90th
# percentile from a dataset of 5000 data points.

def quantile_ci(data, q, c=.95, n_trials=1000):
    """
    Compute a confidence interval for a quantile of a dataset using a bootstrap
    method.

    Input parameters:
        data: data in form of 1-D array-like (e.g. numpy array or Pandas series)
        q: quantile to be estimated, must be between 0 and 1
        c: confidence interval width
        n_trials: number of bootstrap samples to perform

    Output value:
        ci: Tuple indicating lower and upper bounds of bootstrapped
            confidence interval
    """

    n_samples = data.shape[0]

    sample_qs = []
    for _ in range(n_trials):
        sample = np.random.choice(data, n_samples, replace=True)

        sample_q = np.percentile(sample, 100*q)
        sample_qs.append(sample_q)

    upper_limit = np.percentile(sample_qs, q=((1 + c)*100)/2)
    lower_limit = np.percentile(sample_qs, q=((1 - c)*100)/2)

    return (lower_limit, upper_limit)

# reading in the data
data = pd.read_csv(data_dir + 'bootstrapping_data.csv')

plt.hist(data['time'], bins = np.arange(0, data['time'].max()+400, 400))

lims = quantile_ci(data['time'], 0.9)
print(lims)

# Permutation Tests
#-------------------

def quantile_permtest(x, y, q, alternative='less', n_trials=10_000):
    """
    Compute a confidence interval for a quantile of a dataset using a bootstrap
    method.

    Input parameters:
        x: 1-D array-like of data for independent / grouping feature as 0s and 1s
        y: 1-D array-like of data for dependent / output feature
        q: quantile to be estimated, must be between 0 and 1
        alternative: type of test to perform, {'less', 'greater'}
        n_trials: number of permutation trials to perform

    Output value:
        p: estimated p-value of test
    """

    # initialize storage of bootstrapped sample quantiles
    sample_diffs = []

    # For each trial...
    for _ in range(n_trials):
        # randomly permute the grouping labels
        labels = np.random.permutation(y)

        # compute the difference in quantiles
        cond_q = np.percentile(x[labels == 0], 100 * q)
        exp_q = np.percentile(x[labels == 1], 100 * q)

        # and add the value to the list of sampled differences
        sample_diffs.append(exp_q - cond_q)

    # compute observed statistic
    cond_q = np.percentile(x[y == 0], 100 * q)
    exp_q = np.percentile(x[y == 1], 100 * q)
    obs_diff = exp_q - cond_q

    # compute a p-value
    if alternative == 'less':
        hits = (sample_diffs <= obs_diff).sum()
    elif alternative == 'greater':
        hits = (sample_diffs >= obs_diff).sum()

    return (hits / n_trials)



data = pd.read_csv(data_dir + 'permutation_data.csv')
bin_borders = np.arange(0, data['time'].max()+400, 400)
plt.hist(data[data['condition'] == 0]['time'], alpha = 0.5, bins = bin_borders)
plt.hist(data[data['condition'] == 1]['time'], alpha = 0.5, bins = bin_borders)
plt.legend(labels = ['control', 'experiment']);

print(np.percentile(data[data['condition'] == 0]['time'], 90),
      np.percentile(data[data['condition'] == 1]['time'], 90))

quantile_permtest(data['time'], data['condition'], 0.9, alternative = 'less')









