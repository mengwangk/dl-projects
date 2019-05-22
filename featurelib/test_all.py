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
from .ft_nstats import NStatsFt


@pytest.fixture(scope="session")
def df():
    file_path = Path('datasets/lotto/4D.zip')
    return pd.read_csv(file_path, header=0, sep=',', quotechar='"')


def test_nstats(df):
    nstats_ft = NStatsFt(df)
    df_transformed = nstats_ft.transpose_(id_vars=["DrawNo", "DrawDate"], var_name="PrizeType", value_name="LuckyNo")
