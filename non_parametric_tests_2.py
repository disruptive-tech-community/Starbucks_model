import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
data_dir = './data/'

# Rank-Sum Test (Mann-Whitney)

def ranked_sum(x, y, alternative='two-sided'):
    """
    Return a p-value for a ranked-sum test, assuming no ties.

    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}

    Output value:
        p: estimated p-value of test
    """

    # compute U
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties

    # compute a z-score
    n_1 = x.shape[0]
    n_2 = y.shape[0]
    mean_u = n_1 * n_2 / 2
    sd_u = np.sqrt(n_1 * n_2 * (n_1 + n_2 + 1) / 12)
    z = (u - mean_u) / sd_u

    # compute a p-value
    if alternative == 'two-sided':
        p = 2 * stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p = stats.norm.cdf(z)
    elif alternative == 'greater':
        p = stats.norm.cdf(-z)

    return p

data = pd.read_csv(data_dir + 'permutation_data.csv')
data.head()

# data visualization
bin_borders = np.arange(0, data['time'].max()+400, 400)
plt.hist(data[data['condition'] == 0]['time'], alpha = 0.5, bins = bin_borders)
plt.hist(data[data['condition'] == 1]['time'], alpha = 0.5, bins = bin_borders)
plt.legend(labels = ['control', 'experiment']);

ranked_sum(data[data['condition'] == 0]['time'],data[data['condition'] == 1]['time'],alternative = 'greater')

# Mann Whitney U as implemented by scipy package
stats.mannwhitneyu(data[data['condition'] == 0]['time'],
                   data[data['condition'] == 1]['time'],
                   alternative = 'greater')


def sign_test(x, y, alternative='two-sided'):
    """
    Return a p-value for a ranked-sum test, assuming no ties.

    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}

    Output value:
        p: estimated p-value of test
    """

    # compute parameters
    n = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p = min(1, 2 * stats.binom(n, 0.5).cdf(min(k, n - k)))
    if alternative == 'less':
        p = stats.binom(n, 0.5).cdf(k)
    elif alternative == 'greater':
        p = stats.binom(n, 0.5).cdf(n - k)

    return p

data = pd.read_csv(data_dir + 'signtest_data.csv')
data.head(10)

# data visualization
plt.plot(data['day'], data['control'])
plt.plot(data['day'], data['exp'])
plt.legend()

plt.xlabel('Day of Experiment')
plt.ylabel('Success rate');

sign_test(data['control'], data['exp'], 'less')