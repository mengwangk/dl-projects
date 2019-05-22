#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Number statistics.

"""
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pandas import DataFrame
from .core import FtAbstract


@dataclass()
class NStatsFt():
    """Number statistical feature.
    """
    data: DataFrame

    def test(self):
        print('testing')
