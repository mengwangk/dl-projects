#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
featurelib
==========
Number statistics.

"""

from dataclasses import dataclass
from .core import FeatureAbstract


@dataclass()
class NumberStatsFeature(FeatureAbstract):
    """Number statistical features.
    """

    def print(self):
        print('testing')
