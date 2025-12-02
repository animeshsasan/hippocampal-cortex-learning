from typing import Tuple
from scipy.stats import bootstrap # type: ignore[import]
import numpy as np

def get_bootstrapped_value(data) -> Tuple[float, float]:
    # 2. Define the statistic to be bootstrapped (e.g., the mean)
    def statistic(data):
        return np.mean(data)

    # 3. Perform the bootstrap
    #   - data: the original dataset, wrapped in a tuple for the function
    #   - statistic: the function to compute the statistic
    #   - n_resamples: number of bootstrap replicates
    #   - confidence_level: desired confidence level (e.g., 0.95 for 95%)
    #   - method: 'percentile' is a common and straightforward method
    res = bootstrap((data,), statistic, n_resamples=100,
                    confidence_level=0.95, method='percentile')

    # 4. Extract the confidence interval
    lower_bound = res.confidence_interval.low
    upper_bound = res.confidence_interval.high
    return lower_bound, upper_bound