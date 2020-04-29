#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:34:06 2020

Convergence data plotted

@author: jeg
"""

import math as mt
import numpy as np
import matplotlib.pyplot as plt

# Read in the text file
fname = 'python results/convergence010_smooth.txt'
con_data = np.loadtxt(fname)