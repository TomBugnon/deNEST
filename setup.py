#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

# pylint: disable=exec-used,invalid-name

from os.path import exists

from setuptools import setup, find_packages

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
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    license=about["__license__"],
    long_description=open('README.md').read() if exists('README.md') else "",
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    packages=find_packages(exclude=["docs", "test"]),
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
    ],
    project_urls={
        "Bug Reports": "https://github.com/tombugnon/denest/issues",
        "Documentation": "https://denest.readthedocs.io",
    },
)
