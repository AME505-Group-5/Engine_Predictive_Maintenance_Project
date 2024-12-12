#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:28:36 2024

@author: tonyshara
"""

from plot_helpers import plot_lpf_summary, plot_fft
from preprocessing_helpers import butter_lowpass
import numpy as np

cutoff_low=5
fs=1000
order=5


