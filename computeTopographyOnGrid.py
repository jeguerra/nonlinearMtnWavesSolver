#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:43:05 2019

@author: TempestGuerra
"""
import sys, getopt
import numpy as np
from numpy import multiply as mul
from scipy import linalg as las
import math as mt

def computeTopographyOnGrid(REFS, profile, opt):
       
       # Get data from REFS
       x = REFS[0]
       
       # Evaluate the function with different options
       if profile == 1:
              # Witch of Agnesi here
              h0 = opt[0]
              aC = opt[1]
       elif profile == 2:
              # Schar mountain
              h0 = opt[0]
              aC = opt[1]
              lC = opt[2]
              # Compute the height field
              ht1 = h0 * np.exp(-1.0 / aC**2.0 * np.power(x, 2.0))
              ht2 = np.power(np.cos(mt.pi / lC * x), 2.0);
              ht = mul(ht1, ht2)
              # Compute the slope field perfectly
              dht1 = -ht1
              dht2 = (2.0 / aC**2.0) * x
              dht3 = ht2
              dht4 = (mt.pi / lC) * np.sin(2.0 * mt.pi / lC * x)
              dhdx = mul(dht2, dht3)
              dhdx = np.add(dhdx, dht4)
              dhdx = mul(dht1, dhdx)
              #dhdx = -DS.hC * exp(-xlon.^2/DS.aC^2) .* ( ...
              #2.0 * xlon / (DS.aC^2) .* (cos(pi * xlon / DS.lC)).^2 + ...
              #pi/DS.lC * sin(2.0 * pi * xlon / DS.lC));
       elif profile == 3:
              # General even power exponential times a cosine series
              h0 = opt[0]
       elif profile == 4:
              # General even power exponential times a polynomial series
              h0 = opt[0]
       elif profile == 5:
              # Terrain data input from a file, maximum elevation set in opt[0]
              h0 = opt[0]
       else:
              print('ERROR: invalid terrain option.')
              sys.exit(2)
              
       # Check that terrain data is positive definite...
       
       return ht, dhdx
              