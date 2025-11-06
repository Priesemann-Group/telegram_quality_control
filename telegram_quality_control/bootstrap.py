import numpy as np
import pandas as pd
import polars as pl

from tqdm import tqdm

from typing import Callable

from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt

# ======================================================================================
# Different functions to compute CI intervals using bootstrapping. Most functions
# assume that we are dealing with a pandas DataFrame.


def bootstrap_CI(
    data: pd.DataFrame | pd.Series | pl.DataFrame | pl.Series,
    func: Callable,
    *args,
    quantile=0.95,
    num_workers=1,
    num_bootstrap=1000,
    **kwargs,
):
    """
    Perform bootstrap to get CI intervals. Executes the function
    ```
        func(data, *args, **kwargs)
    ```
    on a subsample of `data`. For each subsample, the same number of rows are drawn with replacement.

    Args:
        data: DataFrame with the dataset
        func: function that computes the statistics
        n_bootstrap: number of bootstrap samples
        args, kwargs - additional arguments to pass to func.
    """

    # Helper function for one sample
    def single_bootstrap(data, func, *args, **kwargs):
        sample_size = len(data)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            bootstrap_sample = data.sample(n=sample_size, replace=True)
        elif isinstance(data, (pl.DataFrame, pl.Series)):
            bootstrap_sample = data.sample(n=sample_size, with_replacement=True)

        # Compute statistic on bootstrap sample
        return func(bootstrap_sample, *args, **kwargs)

    lower_bound, upper_bound = _abstract_bootstrap_CI(
        single_bootstrap,
        data,
        func,
        *args,
        quantile=quantile,
        num_workers=num_workers,
        num_bootstrap=num_bootstrap,
        **kwargs,
    )

    return lower_bound, upper_bound


def cluster_bootstrap_CI(
    data: pd.DataFrame | pl.DataFrame,
    func: Callable,
    cluster_col: str,
    *args,
    quantile=0.95,
    num_workers=1,
    num_bootstrap=1000,
    **kwargs,
):
    """
    Perform cluster bootstrap to get CI intervals. Executes the function
    ```
        func(data, *args, **kwargs)
    ```
    on a subsample of `data`. For each subsample, clusters are drawn with replacement. Then, all
    rows that correspond to a cluster are added to a subsample.

    Parameters:
    - data: DataFrame with the dataset
    - func: function that computes the statistics
    - cluster_col: column defining clusters. Should be a categorical value that supports comparison
        with `=` (not floats).
    - n_bootstrap: number of bootstrap samples
    - args, kwargs - additional arguments to pass to func.
    """

    # Helper function for one sample
    def single_bootstrap(data, func, cluster_col, *args, **kwargs):
        # Get unique clusters
        unique_clusters = data[cluster_col].unique()
        n_clusters = len(unique_clusters)

        # Resample clusters with replacement
        sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)

        # Reconstruct dataset from sampled clusters
        if isinstance(data, (pd.DataFrame, pd.Series)):
            bootstrap_sample = pd.concat(
                [data[data[cluster_col] == cluster] for cluster in sampled_clusters],
                ignore_index=True,
            )
        elif isinstance(data, (pl.DataFrame, pl.Series)):
            bootstrap_sample = pl.concat(
                [data.filter(pl.col(cluster_col) == cluster) for cluster in sampled_clusters]
            )

        # Compute statistic on bootstrap sample
        return func(bootstrap_sample, *args, **kwargs)

    lower_bound, upper_bound = _abstract_bootstrap_CI(
        single_bootstrap,
        data,
        func,
        cluster_col,
        *args,
        quantile=quantile,
        num_workers=num_workers,
        num_bootstrap=num_bootstrap,
        **kwargs,
    )

    return lower_bound, upper_bound


def _abstract_bootstrap_CI(
    single_bootstrap: Callable,
    *args,
    quantile=0.95,
    num_workers=1,
    num_bootstrap=1000,
    **kwargs,
):
    """Helper function that abstracts away the common functionality between normal bootstrap and clustered bootstrap.

    Args:
        single_bootstrap (Callable): a function to compute one bootstrap iteration
    """
    bootstrap_samples = []
    if num_workers == 1:
        # Sequential execution
        for i in tqdm(range(num_bootstrap)):
            sample = single_bootstrap(*args, **kwargs)
            bootstrap_samples.append(sample)

    else:
        # Parallel execution
        bootstrap_samples = Parallel(
            n_jobs=num_workers,
            verbose=10,
            batch_size=1,
        )(delayed(single_bootstrap)(*args, **kwargs) for _ in range(num_bootstrap))

    bootstrap_samples = np.array(bootstrap_samples)

    lower_bound = np.quantile(bootstrap_samples, 0.5 - 0.5 * quantile, axis=0)
    upper_bound = np.quantile(bootstrap_samples, 0.5 + 0.5 * quantile, axis=0)

    return lower_bound, upper_bound
