#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py


"""Plot."""


import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pylab
from bokeh import palettes
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure

TOOLS = 'crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select'
PALETTE = palettes.Inferno256


def show_distribution(image, min_value=None):
    """Show histogram of values greater than ``min_value`` in image or list."""
    if isinstance(image, list):
        flat = list
    elif isinstance(image, (np.ndarray)):
        flat = image.flatten()
    if min_value is not None:
        flat = [x for x in flat if x > min_value]
    pylab.hist(flat)
    pylab.show()


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


def show_average_raster(activity,
                        xmin=None, xmax=None, interpolation='none',
                        plot_cols=None):
    """Show raster plot of activity by column, in the active figure.

    Args:
        activity (np-array): (ntimesteps *nrows * ncols) activity array of a
            population.
        xmin, xmax (float): lower and higher bound for color coding.
        interpolation (str): type of interpolation used in plt.matshow
        plot_cols (list): List of columns to plot one under the other.
            The raster plot has dimensions
            ((ntimesteps * len(plot_cols)) * ntimesteps). If not specified, plot
            all the columns.
    """
    ntimesteps, nrows, ncols = activity.shape

    if plot_cols is None:
        # Plot all the columns
        plot_cols = range(ncols)
    if xmin is None:
        xmin = np.min(activity)
    if xmax is None:
        xmax = np.max(activity)

    # Concatenate the columns to have a ((ntimesteps * len(plot_cols)) *
    # ntimesteps)
    col_activity = np.reshape(activity[:, :, plot_cols],
                              (ntimesteps, nrows * len(plot_cols)),
                              order='F')
    subp = plt.subplot2grid((1, 3), (0, 0), colspan=3)
    subp.matshow(np.transpose(col_activity), interpolation=interpolation,
                      aspect='auto', vmin=xmin, vmax=xmax)
