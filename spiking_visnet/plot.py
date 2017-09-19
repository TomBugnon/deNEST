#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py


"""Plot."""


import itertools
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bokeh import palettes
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure

from .analysis.activity import all_cv, mean_activity
from .save import load_activity

TOOLS = 'crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select'
PALETTE = palettes.Inferno256


def make_activity_figure(pops, plot_period, output_dir, fig_title='figure'):
    """Create raster as sublots for each population/variable.

    Args:
        pops (list): List of tuples each of the form::
            ('layer_name', 'population_name', 'variable_name')
    """
    fig = plt.figure(dpi=100)
    fig.set_size_inches(8.27, 11.69) # Vertical A4

    fig.suptitle(fig_title)

    for i in range(len(pops)):
        pop = pops[i]
        layer = pop[0]
        population = pop[1]
        variable = pop[2]
        activity = load_activity(output_dir, layer, population,
                                 variable=variable)

        subp = fig.add_subplot(len(pops), 1, i+1)
        show_activity_raster(activity[plot_period], plot_cols=[0],
                             variable=variable, subp=subp, show=False)
        subp.set_ylabel(layer + ',\n' + population)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_dir, fig_title))
    plt.show()
    plt.close()


def show_layer_summary(activity, variable, layer, pop, plot_period=None,
                       rate_period=None, cv_period=None, plot_cols=None,
                       show_raster=True, show_rate=True, show_cv=False):
    if layer=='retina':
        variable = 'spikes'
    if variable != 'spikes':
        show_rate = False
        show_cv = False

    if show_raster:
        print('Activity raster plot: ', variable)
        show_activity_raster(activity[plot_period], plot_cols,
                             variable=variable)
    if show_rate:
        print('Firing rate:')
        rates = mean_activity(activity[rate_period], variable='spikes')
        show_mean_min_max(rates)

    if show_cv:
        print('CV distribution:')
        show_distribution(all_cv(activity))


def show_distribution(image, plot_density=True, plot_histogram=True,
                      plot_rugplot=False, nbins=30, show_plot=True, label=None,
                      min_value=None, max_value=None):
    """Show distribution of values in array or list.

    Show histogram, density plot and or rugplot of values between min_value and
    max_value, using seaborn ``distplot`` function.
    """
    # Flatten
    if isinstance(image, list):
        flat = list
    elif isinstance(image, (np.ndarray)):
        flat = image.flatten()
    else:
        return

    # Filter
    if min_value is not None:
        flat = [x for x in flat if x > min_value]
    if max_value is not None:
        flat = [x for x in flat if x < max_value]

    # Plot
    if len(set(flat)) == 1:
        print('All the values in the filtered array are equal.\n' \
              + '   => Not plotting distribution.')
        return

    sns.set()
    sns.distplot(flat, hist=plot_histogram, bins=nbins, kde=plot_density,
                 rug=plot_rugplot, label=label)

    # Show
    if show_plot:
        plt.legend()
        plt.show()


def show_array_of_images(array_of_images, figsize=(40, 40), plot_rows=None,
                         plot_cols=None):
    """Show array of images in separate subplots.

    Args:
        array_of_images: (nrows_t, ncols_t, nrows_s, nrows_s) np-array. Each
            (row, col, :, :) is an image.
        figsize (tuple): size of the figure,
        plot_rows, plot_cols (list): locations of the images to be plotted.

    """
    fig = plt.figure(figsize=figsize)
    if plot_rows is None:
        plot_rows = range(np.size(array_of_images, 0))
    if plot_cols is None:
        plot_cols = range(np.size(array_of_images, 1))

    nrows = len(plot_rows)
    ncols = len(plot_cols)
    for i, j in itertools.product(range(nrows), range(ncols)):
        plt.subplot2grid((nrows, ncols), (i, j))
        plt.imshow(array_of_images[plot_rows[i], plot_cols[j], :, :])
    fig.show()


def show_im(image):
    """Use pyplot to show a numpy image."""
    plt.imshow(image)
    plt.show()


def show_mean_min_max(array):
    """Print min, max and mean of an array to screen."""
    print('mean: ', np.mean(array))
    print('min: ', np.min(array))
    print('max: ', np.max(array), '\n')

def init():
    """Initialize bokeh figure with jupyter notebook."""
    plot = figure(tools=TOOLS)
    output_notebook()
    return plot


def color(a):
    """TODO."""
    return [PALETTE[max(min(int(i), 255), 0)] for i in a]


def animate(plot, movie, fps=5, t=0, T=None, size=1):
    """Plot the frames t to T of movie in an animated bokeh plot.

    Args:
        - plot (bokeh Figure)
        - movie (np.array): (time * nrows * ncols)
        - fps: frames per second
        - t: index of the first frame
        - T: index of the last frame (default: number of frames in movie).

    """
    nframes, nrows, ncols = np.shape(movie)
    print(np.shape(movie))
    N = nrows * ncols

    if T is None:
        T = nframes

    # Format movie in N*T instead of T*nrows*ncols
    mv_1d = np.reshape(movie, (T, N))
    # Scale by length of palette
    mv_1d = mv_1d * len(PALETTE)
    # row, col positions for each of the pixels in the '1D image'
    positions = np.array(list(itertools.product(range(nrows), range(ncols))))
    # Convert values to colors
    color_mv_1d = [color(mv_1d[t, :]) for t in range(T)]

    # Initialize plot
    r = plot.circle(positions[::1, 1],  # NB mathematical coordinate, not numpy
                    positions[::-1, 0],
                    fill_color=np.reshape(movie[t, :, :], (N,)),
                    radius=0.5,
                    color='white')

    # Target of the plot (notebook)
    target = show(plot, notebook_handle=True)

    while t < T:
        plot.title.text = f'frame # = {t}'
        r.data_source.data['fill_color'] = color_mv_1d[t]
        # Push updates to the plot continuously using the handle
        # (interrupt the notebook kernel to stop)
        push_notebook(handle=target)
        t += 1
        time.sleep(1 / fps)


def show_activity_raster(activity, plot_cols=None, figsize=(40, 10), xmin=None,
                         xmax=None, interpolation='none', variable='spikes',
                         fig=None, subp=None, show=False):
    """Show raster plot of activity by column, in the active figure.

    Args:
        activity (np-array): (ntimesteps *nrows * ncols) activity array of a
            population.
        plot_cols (list): List of columns to plot one under the other.
            The raster plot has dimensions
            ((ntimesteps * len(plot_cols)) * ntimesteps). If not specified, plot
            all the columns.
        figsize (tuple): Size of figure.
        xmin, xmax (float): lower and higher bound for color coding.
        interpolation (str): type of interpolation used in plt.matshow
        variable (str): If 'V_m', default (xmin, xmax) = -70, -45
    """
    if fig is None and subp is None:
        fig = plt.figure(figsize=figsize)
        subp = plt.subplot2grid((1, 3), (0, 0), colspan=3)

    ntimesteps, nrows, ncols = activity.shape

    if plot_cols is None:
        # Plot all the columns
        plot_cols = range(ncols)
    if xmin is None:
        xmin = np.min(activity) if variable == 'spikes' else -70
    if xmax is None:
        xmax = np.max(activity) if variable == 'spikes' else -45

    # Concatenate the columns to have a ((ntimesteps * len(plot_cols)) *
    # ntimesteps)
    col_activity = np.reshape(activity[:, :, plot_cols],
                              (ntimesteps, nrows * len(plot_cols)),
                              order='F')
    ax = subp.matshow(np.transpose(col_activity), interpolation=interpolation,
                      aspect='auto', vmin=xmin, vmax=xmax)
    if show:
        plt.show()
        plt.close()

    return fig, subp, ax
