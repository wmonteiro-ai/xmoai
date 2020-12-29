#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:18:52 2020

@author: wellington
"""


import setuptools
from xmoai import __version__

__name__ = 'xmoai'
__author__ = 'Wellington R Monteiro'
__url__ = 'https://github.com/wmonteiro92/xmoai'

with open("README.md", "r") as fh:
    print(__version__)
    long_description = fh.read()

setuptools.setup(
    name=__name__, # Replace with your own username
    version=__version__,
    author=__author__,
    author_email="wellington.r.monteiro@gmail.com",
    description="eXplainable Artificial Intelligence using Multiobjective Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['pymoo>=0.3.0',
                      'numpy>=1.16.0',
                      'gower>=0.0.5',
                      'topsis-jamesfallon>=0.2.3']
)
