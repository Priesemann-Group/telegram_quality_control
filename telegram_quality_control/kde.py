import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt

# ======================================================================================
# Functions to calculate kernel density estimations from scattered data.
# https://en.wikipedia.org/wiki/Kernel_density_estimation


def kde_fit(data, bandwidth=0.05, num_samples=100, xmin=0, xmax=1, cluster_ids=None):
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
    weight_matrix /= weight_matrix.sum().sum()

    prob = weight_matrix.sum(axis=1)

    prob = prob / np.sum(prob)  # Normalize the probabilities

    # calculate the std of the distribution that one would get under bootstrapping
    if cluster_ids is None:
        # Var(f_KDE) = sum(W_ij^2) - 1 / N * (sum(W_ij))^2
        var_kde = (weight_matrix**2).sum(axis=1) - 1 / weight_matrix.shape[1] * prob**2
        std_kde = np.sqrt(var_kde)
    else:
        # m - category, M - num categories
        # Var(f_KDE) = sum_i1 sum_i2 where m(i1) == m(i2) W_(j, i1) W_(j, i2) - 1 / M * (sum(W_ij))^2
        unique_clusters = np.unique(cluster_ids)
        num_clusters = len(np.unique(cluster_ids))
        var_kde = np.zeros(num_samples)

        # Sort the columns of the weight matrix by clusters
        weight_matrix_sorted = weight_matrix.copy()
        sortids = np.argsort(cluster_ids)
        weight_matrix_sorted = weight_matrix_sorted[:, sortids]
        cluster_ids = cluster_ids[sortids]

        for id in unique_clusters:
            mask = cluster_ids == id
            var_kde += np.sum(weight_matrix_sorted[:, mask], axis=1) ** 2

        var_kde -= 1 / num_clusters * prob**2
        std_kde = np.sqrt(var_kde)

    x_points = x_points.flatten()

    return x_points, prob, std_kde


def kde_fit_streaming(data: pl.LazyFrame, column, **kwargs):

    def wrapper(df):
        result = kde_fit(df[column], bandwidth, **kwargs)

    data.map_batches(wrapper, streamable=True)


def boolean_conditional_kde(
    data: pd.DataFrame | pl.DataFrame,
    boolean,
    parameter,
    bandwidth=0.05,
    num_samples=100,
    xmin=0,
    xmax=1,
    cluster_column=None,
):
    """
    Returns the probability that a boolean random variable is True given a condition.
    P(boolean=True | condition)

    data: DataFrame, the data containing the boolean and condition columns
    boolean: str, column name of the boolean variable
    condition: str, column name of the condition variable
    """

    if cluster_column is not None:
        cluster_ids = data[cluster_column]
    else:
        cluster_ids = None

    cond_mask = data[boolean]

    # P(condition | boolean)
    if type(data) is pd.DataFrame:
        if cluster_column is not None:
            cond_cluster_ids = cluster_ids[cond_mask]
        else:
            cond_cluster_ids = None

        masked_data = data[cond_mask]

    elif type(data) is pl.DataFrame:
        if cluster_column is not None:
            cond_cluster_ids = cluster_ids.filter(pl.col(boolean))
        else:
            cond_cluster_ids = None

        masked_data = data.filter(pl.col(boolean))

    x_points, likelihood, likelihood_std = kde_fit(
        masked_data[parameter],
        bandwidth=bandwidth,
        cluster_ids=cond_cluster_ids,
    )

    # P(boolean, condition) = P(condition | boolean) * P(boolean)
    frac_true = data[boolean].mean()
    joint_prob = likelihood * frac_true / likelihood.sum()

    # P(condition)
    _, marginal_prob, marginal_prob_std = kde_fit(
        data[parameter],
        bandwidth=bandwidth,
        cluster_ids=cluster_ids,
    )

    # P(boolean | condition) = P(boolean, condition) / P(condition)
    conditional_prob = joint_prob / marginal_prob

    # simple error propagation
    conditional_prob_std = (frac_true / marginal_prob * likelihood_std) ** 2 + (
        joint_prob / marginal_prob**2 * marginal_prob_std
    ) ** 2
    conditional_prob_std = np.sqrt(conditional_prob_std)

    x_points = x_points.flatten()

    return x_points, conditional_prob, conditional_prob_std


def kde_fit_2D(datapoints, bandwidth, xlims, ylims, num_grid_points=30):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.vstack(datapoints).T)

    # create a grid to plot the density
    x_points = np.linspace(*xlims, num_grid_points)
    y_points = np.linspace(*ylims, num_grid_points)
    x_grid, y_grid = np.meshgrid(x_points, y_points)
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(x_grid.shape)
    return x_grid, y_grid, density


def plot_kde_2D(ax, x_grid, y_grid, density, cmap, levels=20):
    cmap = plt.get_cmap(cmap)
    cmap.set_under('white')
    data_range = density.max() - density.min()
    norm = mpl.colors.BoundaryNorm(
        np.linspace(density.min() + 0.05 * data_range, density.max(), 100), cmap.N
    )

    contour_plot = ax.contourf(
        x_grid, y_grid, density, levels=levels, cmap=cmap, norm=norm, extend='min'
    )
    contour_plot.set_zorder(-1)

    return contour_plot
