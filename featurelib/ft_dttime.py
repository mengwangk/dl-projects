#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Date time feature.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import re 

from pandas import DataFrame
from dataclasses import dataclass

from .core import FtAbstract


@dataclass()
class DateTimeFt(FtAbstract):
    """Date time manipulation"""

    def add_datepart(self, df: DataFrame, fldname: str, drop=True, time=False):
        """Helper function that adds columns relevant to a date.

        Arguments:
            df {DataFrame} -- Pandas data frame.
            fldname {string} -- Field name.

        Keyword Arguments:
            drop {bool} -- Drop the date field (default: {True})
            time {bool} -- Time part (default: {False})
        """
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time:
            attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop:
            df.drop(fldname, axis=1, inplace=True)
