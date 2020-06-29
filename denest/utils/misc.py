#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# utils/misc.py
"""Miscellaneous utils."""

# pylint:=missing-docstring

import time
from pathlib import Path

from ..io import save


def pretty_time(start_time):
    minutes, seconds = divmod(time.time() - start_time, 60)
    hours, minutes = divmod(minutes, 60)
    return "%dh:%02dm:%02ds" % (hours, minutes, seconds)


def drop_versions(output_dir):
    from ..__about__ import __version__
    import nest

    path = Path(
        save.output_subdir(output_dir, "versions"),
        save.output_filename("versions")
    )
    with path.open("w") as f:
        f.write(
            f'denest={__version__}\n'
            f'{nest.version()}\n'
        )
