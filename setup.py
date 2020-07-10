#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name="exspect",
    url="https://github.com/bd-j/exspect",
    version="0.0",
    author="",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["exspect",],
    license="LICENSE",
    description="Prospector paper demos",
    #long_description=open("README.md").read(),
    #package_data={"": ["README.md", "LICENSE"]},
    #scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy", "astropy"],
)
