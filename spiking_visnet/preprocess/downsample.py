#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# downsample.py


# TODO
def downsample(input_movie, preprocessing_params, network):
    return input_movie


def get_string(_, network):
    (xdim, ydim) = network.input_res()
    return ('res_' + str(xdim) + 'x' + str(ydim) + '_')
