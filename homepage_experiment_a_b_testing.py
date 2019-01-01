import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats import proportion as proptests

import matplotlib.pyplot as plt

data_dir = './data/'
# In the dataset, the 'condition' column takes a 0 for the control group, and 1
# for the experimental group. The 'click' column takes a values of 0 for no
# click, and 1 for a click.
data = pd.read_csv(data_dir+'homepage-experiment-data.csv')

# Need to check both invariant metric and evaluation metric
# First check the invariant metric
n_obs = (data['Control Cookies'] + data['Experiment Cookies']).sum()
n_control = data['Control Cookies'].sum()
n_exper = data['Experiment Cookies'].sum()
p = 0.5
q = 1 - p

# Simulation method (lets take 1000000 as the number of trials)
samples = np.random.binomial(n_obs, p, 1000000)
print(np.logical_or(samples <= n_control, samples >= n_exper).sum() / 1000000)


# Analytical method
# calculate the z-score
mean_ = n_obs * p
std_dev = np.sqrt(p * q * n_obs)

z = ((n_control + 0.5) - mean_)/std_dev
print(2 * stats.norm.cdf(z))


# Evaluating metrics
# Download rate - Total Number of downloads / Number of cookies

p_null = ((data['Control Downloads'] + data['Experiment Downloads']).sum(
))/((data['Control Cookies'] + data['Experiment Cookies']).sum())

p_control = (data['Control Downloads'].sum())/(data['Control Cookies'].sum())
p_exper = (data['Experiment Downloads'].sum())/(data['Experiment Cookies'].sum())
q_p_null = 1 - p_null
n_control = data['Control Cookies'].sum()
n_exper = data['Experiment Cookies'].sum()

std_error_sampling_distribution_mean_prop = np.sqrt(p_null *
    q_p_null * ((n_exper + n_control)/(n_control * n_exper)))

z_score = (p_control - p_exper)/std_error_sampling_distribution_mean_prop
stats.norm.cdf(z_score)


# Evaluating metrics
# Download rate - Total Number of licenses / Number of cookies
n_control = data['Control Cookies'][:21].sum()
n_exper = data['Experiment Cookies'][:21].sum()

p_null = ((data['Control Licenses'] + data['Experiment Licenses']).sum(
))/(n_control + n_exper)
q_p_null = 1 - p_null
p_control = (data['Control Licenses'].sum())/n_control
p_exper = (data['Experiment Licenses'].sum())/n_exper

std_error_sampling_distribution_mean_prop = np.sqrt(p_null *
    q_p_null * ((n_exper + n_control)/(n_control * n_exper)))

z_score = (p_control - p_exper)/std_error_sampling_distribution_mean_prop
stats.norm.cdf(z_score)

