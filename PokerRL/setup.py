# Copyright (c) 2019 Eric Steinberger


import os.path as osp

import setuptools


with open('%s/%s' % (osp.dirname(osp.realpath(__file__)), 'requirements.txt')) as f:
    requirements = [line.strip() for line in f]

setuptools.setup(
    name="PokerRL",
    version="0.0.3",
    author="Eric Steinberger",
    author_email="ericsteinberger.est@gmail.com",
    description="A framework for Reinforcement Learning in Poker.",
    license='MIT',
    url="https://github.com/TinkeringCode/PokerRL",
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)
