#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Categorical features.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import re

from pandas import DataFrame
from dataclasses import dataclass

from .core import FtAbstract


@dataclass()
class CategoryFt(FtAbstract):
    pass
