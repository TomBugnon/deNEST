#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_full_simulation.py

"""Regression testing of the full simulation outputs."""

import pickle

import pytest

import nets
import nets.io.load

PARAMS_PATH = './params/default.yml'
INPUT_DIR = './input'
OUTPUT_DIR = './output'


@pytest.fixture(scope='module')
def output_dir():
    nets.run(PARAMS_PATH, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    return OUTPUT_DIR


@pytest.fixture(scope='module')
def metadata_paths(output_dir):
    return nets.io.load.metadata_paths(
        output_dir,
    )


def test_parameter_tree(output_dir, data_regression):
    params_tree = nets.io.load.load_yaml(
        nets.io.load.output_path(output_dir, 'tree')
    )
    data_regression.check(params_tree)


def test_session_times(output_dir, data_regression):
    session_times = nets.io.load.load_session_times(output_dir)
    data_regression.check(session_times)


def test_recorder_metadata(metadata_paths, data_regression):
    # Load metadata for all recorders
    all_metadatas = {}
    for metadata_path in metadata_paths:
        metadata = nets.io.load.load_yaml(metadata_path)
        all_metadatas[str(metadata_path)] = metadata
    # Compare all
    data_regression.check(all_metadatas)


def test_data(metadata_paths, file_regression):
    all_datas = {}
    for metadata_path in metadata_paths:
        recorder_data = nets.io.load.load(metadata_path)
        # Sort dataframe
        all_datas[str(metadata_path)] = recorder_data.sort_values(
            list(recorder_data.columns)
        ).reset_index(drop=True)
    file_regression.check(pickle.dumps(all_datas), binary=True, extension='')
