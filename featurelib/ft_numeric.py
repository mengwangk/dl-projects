#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Numeric features.

"""
import numpy as np
import pandas as pd
import statistics as st
import statsmodels.api as sm

from dataclasses import dataclass
from pandas import DataFrame
from .core import FtAbstract


@dataclass()
class NumericFt(FtAbstract):
    """Numeric features.
    """

    def to_category(self, column_names:list = None, length:int = 0, padding:str = ''):
        """Convert numeric to category column"""
        self.data["LuckyNo"] = self.data["LuckyNo"].str.pad(4, side='left', fillchar='-')
