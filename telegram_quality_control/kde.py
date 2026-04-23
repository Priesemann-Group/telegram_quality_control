import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from typing import Callable

from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt

# ======================================================================================
# Functions to calculate kernel density estimations from scattered data.
# https://en.wikipedia.org/wiki/Kernel_density_estimation


def kde_fit(data, bandwidth=0.05, num_samples=100, xmin=0, xmax=1, cluster_ids=None, cluster_weight_map=None):
    """
    Gaussian Kernel Density Estimation (KDE) for a given 1D data array.
    """

    if isinstance(data, (pd.Series)):
        data = data.values
    elif isinstance(data, pl.Series):
        data = data.to_numpy()

    if isinstance(cluster_ids, pd.Series):
        cluster_ids = cluster_ids.values
    elif isinstance(cluster_ids, pl.Series):
        cluster_ids = cluster_ids.to_numpy()

    mask = ~np.isnan(data)
    data = data[mask]

    if cluster_ids is not None:
        cluster_ids = cluster_ids[mask]

    # Transform the samples to a row vector
    X = data.reshape(1, -1)

    # Transform the points at which to evaluate the KDE to a column vector
    x_points = np.linspace(xmin, xmax, num_samples).reshape(-1, 1)

    sigma = bandwidth

    weight_matrix = (
        1 / np.sqrt(sigma**2 * 2 * np.pi) * np.exp(-((x_points - X) ** 2) / (2 * sigma**2))
    )
    
    if (cluster_ids is not None) and (cluster_weight_map is not None):
        cluster_weights = np.array([cluster_weight_map[id] for id in cluster_ids])
        
        # weigh points according to their cluster weights
        weight_matrix = weight_matrix * cluster_weights
    
    weight_matrix /= weight_matrix.sum().sum()

    prob = weight_matrix.sum(axis=1)

    prob = prob / np.sum(prob)  # Normalize the probabilities

    x_points = x_points.flatten()

    return x_points, prob


def kde_fit_clustered_bootstrap_CI(
    data: pd.DataFrame | pl.DataFrame,
    value_col: str,
    cluster_col: str,
    *args,
    quantile=0.95,
    num_workers=1,
    num_bootstrap=1000,
    **kwargs,
):
    # Helper function for one sample
    def single_bootstrap(data, *args, seed=42, **kwargs):
        # Get unique clusters
        cluster_ids = data[cluster_col]
        unique_clusters = cluster_ids.unique()
        n_clusters = len(unique_clusters)
        
        rng = np.random.default_rng(seed)

        # Resample clusters with replacement
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        
        weight_map = {}
        
        for cluster in unique_clusters:
            weight_map[cluster] = np.sum(sampled_clusters == cluster)
        
        kwargs = {"cluster_ids": cluster_ids, "cluster_weight_map": weight_map} | kwargs
        
        _, prob = kde_fit(data[value_col], *args, **kwargs)
        
        return prob

    lower_bound, upper_bound = _abstract_bootstrap_CI(
        single_bootstrap,
        data,
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
        )(delayed(single_bootstrap)(*args, seed=i, **kwargs) for i in range(num_bootstrap))
    
    bootstrap_samples = np.array(bootstrap_samples)

    lower_bound = np.quantile(bootstrap_samples, 0.5 - 0.5 * quantile, axis=0)
    upper_bound = np.quantile(bootstrap_samples, 0.5 + 0.5 * quantile, axis=0)

    return lower_bound, upper_bound
