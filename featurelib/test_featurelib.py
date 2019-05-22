#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-


import pytest

from .ft_nstats import NStatsFt

def test_nstats():
    data = None
    nstats_ft = NStatsFt(data)
    nstats_ft.test()
    
