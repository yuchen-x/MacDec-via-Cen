#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

setup(
    name='rlmamr',
    version='0.0.1',
    description='rlmamr - reinforcement learning for macro-action multi-robot',
    author='Yuchen Xiao',
    author_email='xiao.yuch@husky.neu.edu',
    url='https://github.com/yuchen-x/MacDec-via-Cen',

    packages=['rlmamr'],
    package_dir={'': 'src'},

    # TODO setup dependencies correctly!

    scripts=[
        'scripts/ma_hddrqn.py',
        'scripts/ma_cen_condi_ddrqn.py',
        'scripts/ma_dec_cen_hddrqn.py',
        'scripts/ma_dec_cen_hddrqn_sep.py',
    ],

    license='MIT',
)
