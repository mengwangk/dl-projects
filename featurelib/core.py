#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Core functions and classes.

"""
import pandas as pd
import warnings

from pandas import DataFrame
from warnings import warn
from dataclasses import dataclass

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@dataclass()
class FtAbstract():
    """Feature base class.
    """

    _data: DataFrame = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def transpose_(self, id_vars=None, value_vars=None,
                   var_name=None, value_name='value'):
        """Transpose the data frame using Pandas built-in pandas.melt method.
        """
        self.data = self._data.melt(id_vars=id_vars, value_vars=value_vars,
                                    var_name=var_name, value_name=value_name)
        return self._data
