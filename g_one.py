import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats import proportion as proptests

import matplotlib.pyplot as plt

data_dir = './data/'
# In the dataset, the 'condition' column takes a 0 for the control group, and 1
# for the experimental group. The 'click' column takes a values of 0 for no
# click, and 1 for a click.
data = pd.read_csv(data_dir+'statistical_significance_data.csv')

# Need to check both invariant metric and evaluation metric
# First check the invariant metric

# Assuming that the user has exactly 50% chance(null hypotheisis) of being
# allocated to either control group or experimental group, lets check if
# n_control and n_exper has any statistically significant difference from
# the null hypotheses. We will be using two tailed test

n_obs = len(data)
n_condition = data.groupby('condition').size()
n_control = n_condition[0]
n_exper = n_condition[1]
p = 0.5
q = 1 - p


# Simulation method (lets take 200000 as the number of trials)
samples = np.random.binomial(n_obs, p, 200000)

print(np.logical_or(samples <= n_control, samples >= n_exper).sum() / 200000)


# Analytical method
# calculate the z-score
mean_ = n_obs * p
std_dev = np.sqrt(p * q * n_obs)

z = ((491 + 0.5) - mean_)/std_dev
print(2 * stats.norm.cdf(z))

# The invariant metric has exactly 61.27% chances of confirming to the null
# hypothesis. Since there is no statistically significant evidence to reject
#  the null hypothesis will accept the null hypothesis.

# Evaluation matrix

# Analytic method.

prop_ = data.groupby('condition').mean()['click']
p_control = prop_[0]
p_exper = prop_[1]
p_null =  sum(data.click)/n_obs
q_p_null = 1 - p_null

std_error_sampling_distribution_mean_prop = np.sqrt(p_null *
    q_p_null * ((n_exper + n_control)/(n_control * n_exper)))

z_score = (p_control - p_exper)/std_error_sampling_distribution_mean_prop
stats.norm.cdf(z_score)

# Simulation Approach
control_clicks = np.random.binomial(n_control, p_null, 200000)
experiment_clicks = np.random.binomial(n_exper, p_null, 200000)

samples = (experiment_clicks/n_exper) - (control_clicks/n_control)

print((samples >= (p_exper - p_control)).mean())






