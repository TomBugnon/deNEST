#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/misc.py
"""Miscellaneous utils."""

# pylint:disable=missing-docstring

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
