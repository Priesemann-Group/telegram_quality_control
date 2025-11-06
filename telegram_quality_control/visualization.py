import matplotlib.pyplot as plt


def single_col_figure(height_frac):
    width = 3.3  # inches
    height = width * height_frac
    fig = plt.figure(figsize=(width, height))
    return fig


def double_col_figure(height_frac):
    width = 7.0  # inches
    height = width * height_frac
    fig = plt.figure(figsize=(width, height))
    return fig


def get_color_cycle():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    cycle_colors = prop_cycle.by_key()['color']
    return cycle_colors
