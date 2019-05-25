#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Test cases.

"""
import pytest
import pandas as pd
import numpy as np

from pandas import DataFrame
from pathlib import Path
from .ft_numeric import NumericFt


def pytest_runtest_makereport(item, call):
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item


def pytest_runtest_setup(item):
    if "incremental" in item.keywords:
        previousfailed = getattr(item.parent, "_previousfailed", None)
        if previousfailed is not None:
            pytest.xfail("previous test failed (%s)" % previousfailed.name)


@pytest.fixture(scope="session")
def data():
    file_path = Path('datasets/lotto/4D.zip')
    df = pd.read_csv(file_path, header=0, sep=',', quotechar='"', dtype=str)
    return df


def test_nstats(data):
    """Test number feature.
    Arguments:
        data {DataFrame} -- Test data.
    """

    numericFt = NumericFt(data)
    df = numericFt.transpose_(
        id_vars=["DrawNo", "DrawDate"], var_name="PrizeType", value_name="LuckyNo")
    
    df['_1st_digit'] = df['LuckyNo'].str[0:1]
    df['_2st_digit'] = df['LuckyNo'].str[1:2]
    df['_3st_digit'] = df['LuckyNo'].str[2:3]
    df['_4st_digit'] = df['LuckyNo'].str[3:4]

    print(df.columns)
    print(df.tail(23))
