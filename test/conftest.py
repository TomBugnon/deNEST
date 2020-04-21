#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conftest.py

import pytest
from test_layers import BASE_LAYERS, INPUT_LAYERS, init_layer


@pytest.fixture(params=INPUT_LAYERS)
def input_layer(request):
    yield from init_layer(*request.param)


@pytest.fixture(params=BASE_LAYERS)
def base_layer(request):
    yield from init_layer(*request.param)


@pytest.fixture(params=BASE_LAYERS+INPUT_LAYERS)
def layer(request):
    yield from init_layer(*request.param)
