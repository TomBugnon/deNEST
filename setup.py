#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

# pylint: disable=exec-used,invalid-name,missing-docstring

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()

about = {}
with open('./spiking_visnet/__about__.py') as f:
    exec(f.read(), about)

setup(
    name=about['__title__'],
    version=about['__version__'],
    long_description=readme,
    packages=['spiking_visnet'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ]
)
