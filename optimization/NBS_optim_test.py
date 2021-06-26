import numpy as np
import random
import time


def noisy_binary_search(element, start, stop, p, step=1e-1, max_iters=1000, early_termination_width=1):
    # Check if p<0.5
    start_time=time.time()
    if p <= 0.5:
        return -1

    # Initialize Uniform Distribution
    range_idx = np.arange(start, stop, step)
    range_dist = np.ones_like(range_idx)
    range_dist = range_dist / np.sum(range_dist)

    # Make Distribution Logarithmic to handle overflows
    range_dist = np.log(range_dist)

    def get_median(range_dist):
        exp_dist = np.exp(range_dist)
        alpha = exp_dist.sum() * 0.5

        # Finding the median of the distribution requires
        # adding together many very small numbers, so it's not
        # very stable. In part, we address this by randomly
        # approaching the median from below or above.
        if random.choice([True, False]):
            return range_idx[exp_dist.cumsum() < alpha][-1]
        else:
            return range_idx[::-1][exp_dist[::-1].cumsum() < alpha][-1]

    def get_belief_interval(range_dist, fraction=0.95):
        exp_dist = np.exp(range_dist)

        epsilon = 0.5 * (1 - fraction)
        epsilon = exp_dist.sum() * epsilon

        left = range_idx[exp_dist.cumsum() < epsilon][-1]
        right = range_idx[exp_dist.cumsum() > (exp_dist.sum() - epsilon)][0]
        return left, right

    for i in range(max_iters):
        # get Median of Distribution
        median = get_median(range_dist)

        # comparision function
        if element >= median:
            range_dist[range_idx >= median] += np.log(p)
            range_dist[range_idx < median] += np.log(1 - p)
        else:
            range_dist[range_idx <= median] += np.log(p)
            range_dist[range_idx > median] += np.log(1 - p)

        # avoid overflows
        range_dist -= np.max(range_dist)

        belief_start, belief_end = get_belief_interval(range_dist)
        if (belief_end - belief_start) <= early_termination_width:
            break

    print(f" p_val: {round(p,2)}, belief_interval: {belief_start}-{belief_end}, iter_count: {i}, time taken: {time.time() -start_time } secs")

    return belief_start, belief_end


for i in np.arange(0.55, 1, 0.05):
    noisy_binary_search(51, 0, 1000, i, step=1e-1, max_iters=10000, early_termination_width=1e-0)
    noisy_binary_search(51, 0, 1000, i, step=1e-2, max_iters=10000, early_termination_width=1e-1)
    noisy_binary_search(51, 0, 1000, i, step=1e-3, max_iters=10000, early_termination_width=1e-2)
    noisy_binary_search(51, 0, 1000, i, step=1e-4, max_iters=10000, early_termination_width=1e-3)
    # noisy_binary_search(51, 0, 1000, i, step=1e-5, max_iters=10000, early_termination_width=1e-4)
