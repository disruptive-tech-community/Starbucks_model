import numpy as np
from scipy import stats


def peeking_sim(alpha=.05, p=.5, n_trials=1000, n_blocks=2, n_sims=10000):
    """
    This function simulates the rate of Type I errors made if an early
    stopping decision is made based on a significant result when peeking ahead.

    Input parameters:
        alpha: Supposed Type I error rate
        p: Probability of individual trial success
        n_trials: Number of trials in a full experiment
        n_blocks: Number of times data is looked at (including end)
        n_sims: Number of simulated experiments run

    Return:
        p_sig_any: Proportion of simulations significant at any check point,
        p_sig_each: Proportion of simulations significant at each check point
    """

    # generate data
    trials_per_block = np.ceil(n_trials / n_blocks).astype(int)
    data = np.random.binomial(trials_per_block, p, [n_sims, n_blocks])

    # standardize data
    data_cumsum = np.cumsum(data, axis=1)
    block_sizes = trials_per_block * np.arange(1, n_blocks + 1, 1)
    block_means = block_sizes * p
    block_sds = np.sqrt(block_sizes * p * (1 - p))
    data_zscores = (data_cumsum - block_means) / block_sds

    # test outcomes
    z_crit = stats.norm.ppf(1 - alpha / 2)
    sig_flags = np.abs(data_zscores) > z_crit
    temp = sig_flags.sum(axis=1)
    p_sig_any = ( temp> 0).mean()
    p_sig_each = sig_flags.mean(axis=0)

    return (p_sig_any, p_sig_each)

peeking_sim(n_trials = 10_000, n_sims = 100_000)


def peeking_correction(alpha=.05, p=.5, n_trials=1000, n_blocks=2,
                       n_sims=10000):
    """
    This function uses simulations to estimate the individual error rate necessary
    to limit the Type I error rate, if an early stopping decision is made based on
    a significant result when peeking ahead.

    Input parameters:
        alpha: Desired overall Type I error rate
        p: Probability of individual trial success
        n_trials: Number of trials in a full experiment
        n_blocks: Number of times data is looked at (including end)
        n_sims: Number of simulated experiments run

    Return:
        alpha_ind: Individual error rate required to achieve overall error rate
    """

    # generate data
    trials_per_block = np.ceil(n_trials / n_blocks).astype(int)
    data = np.random.binomial(trials_per_block, p, [n_sims, n_blocks])

    # standardize data
    data_cumsum = np.cumsum(data, axis=1)
    block_sizes = trials_per_block * np.arange(1, n_blocks + 1, 1)
    block_means = block_sizes * p
    block_sds = np.sqrt(block_sizes * p * (1 - p))
    data_zscores = (data_cumsum - block_means) / block_sds

    # find necessary individual error rate
    max_zscores = np.abs(data_zscores).max(axis=1)
    z_crit_ind = np.percentile(max_zscores, 100 * (1 - alpha))
    alpha_ind = 2 * (1 - stats.norm.cdf(z_crit_ind))

    return alpha_ind


peeking_correction(n_trials = 10_000, n_sims = 100_000)










