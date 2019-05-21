#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-


import pytest
import unittest

def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5