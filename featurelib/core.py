#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Core functions and classes.

"""
import warnings
from warnings import warn

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


class FtAbstract():
    """Feature abstract class.
    """
    pass