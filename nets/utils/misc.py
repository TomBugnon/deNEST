#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/misc.py
"""Miscellaneous utils."""

import subprocess
import time
from os.path import join

from .. import save


def pretty_time(start_time):
    minutes, seconds = divmod(time.time() - start_time, 60)
    hours, minutes = divmod(minutes, 60)
    return "%dh:%02dm:%02ds" % (hours, minutes, seconds)


def drop_git_hash(output_dir):
    git_hash = git_head_hash()
    path = join(save.output_subdir(output_dir, 'git_hash'),
                save.output_filename('git_hash'))
    with open(path, 'wb') as f:
        f.write(git_hash)


def git_head_hash():
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False,
                               stdout=subprocess.PIPE)
    return process.communicate()[0].strip()


def generate_output_subdir(main_output_dir, label=None,
                           base_output_name='output_'):
    """Return the path to a subdirectory of `main_output_dir` to use as output.

    The returned path has the following structure::
        `main_output_dir`/`base_output_name_`+`N`+ '_' + `date` + '_' + `label`
    where:
        - `N` is incremented at each run depending on the existing
        subdirectories of `main_output_dir`
        - `date` is of the form yyyy/mm/dd/hh:mm
    """
    import datetime
    import os
    import os.path

    # Get N
    # Get the name of all the subdirectories in main_output_dir that start with
    # 'base_output_name'
    all_subdirs = [filename[len(base_output_name):]
                   for filename in os.listdir(main_output_dir)
                   if os.path.isdir(os.path.join(main_output_dir,
                                                 filename))
                   and filename.startswith(base_output_name)]
    print(all_subdirs)
    all_subdirs_N = [int(subdir.split('_')[0]) for subdir in all_subdirs
                     if subdir.split('_')[0].isdigit()]

    N = 1 if not all_subdirs_N else max(all_subdirs_N) + 1

    # Get time string
    timestr = '{date:%Y-%m-%d}'.format(date = datetime.datetime.now())

    # label string
    labelstr = '' if label is None else '_' + label

    output_subdir = base_output_name + str(N) + '_' + timestr + labelstr

    return os.path.join(main_output_dir, output_subdir)
