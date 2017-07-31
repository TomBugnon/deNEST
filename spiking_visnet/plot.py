#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py


"""Plot."""


import itertools
import time

import numpy as np
from bokeh import palettes
from bokeh.io import output_notebook, push_notebook, show
from bokeh.plotting import figure

TOOLS = 'crosshair,pan,wheel_zoom,box_zoom,reset,tap,box_select,lasso_select'
PALETTE = palettes.Inferno256


def show_im(im):
    """Use pyplot to show a numpy image."""
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()


def init():
    """Initialize bokeh figure with jupyter notebook."""
    p = figure(tools=TOOLS)
    output_notebook()
    return p


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
    print(mv_1d.shape)
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
