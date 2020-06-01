#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

# pylint: disable=exec-used,invalid-name

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()

about = {}
with open('./denest/__about__.py') as f:
    exec(f.read(), about)

install_requires = [
    'docopt',
    'numpy',
    'pandas',
    'pyyaml',
    'tqdm',
]

setup(
    name=about['__title__'],
    version=about['__version__'],
    long_description=readme,
    packages=['denest'],
    zip_safe=False,
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
    ]
)
