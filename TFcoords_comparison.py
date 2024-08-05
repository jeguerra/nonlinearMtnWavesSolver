#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 07:16:11 2021

@author: jeg
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt
from computeGrid import computeGrid
import computeDerivativeMatrix as derv
import computeTopographyOnGrid as top
import computeGuellrichDomain2D as tfg

#%%
# Make a test function and its derivative (DEBUG)

 # Set grid dimensions and order
L2 = 50000.0
L1 = -L2
ZH = 20000.0
NX = 383
NZ = 83
DIMS = [L1, L2, ZH, NX, NZ]

h0 = 2000.0
aC = 5000.0
lC = 4000.0
HOPT = [h0, aC, lC, 1.2E+4, False, 2]

# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, True)
  
# Turn on cubic spline derivatives...
DDX_CS, DDX2_CS = derv.computeCubicSplineDerivativeMatrix(DIMS, REFS[0], True)
DDZ_CS, DDX2_CS = derv.computeCubicSplineDerivativeMatrix(DIMS, REFS[1], True)

# Update the REFS collection
REFS.append(DDX_CS)
REFS.append(DDZ_CS)

#% Read in topography profile or compute from analytical function
HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT, DDX_CS)
       
# Make the 2D physical domains from reference grids and topography
zRay = DIMS[2] - 1000.0
# USE THE GUELLRICH TERRAIN DECAY
XL1, ZTL1, DZT1, sigma1, ZRL, DXM, DZM = \
       tfg.computeGuellrichDomain2D(DIMS, REFS, zRay, HofX, dHdX, True)
# USE UNIFORM STRETCHING
XL2, ZTL2, DZT2, sigma2, ZRL = tfg.computeStretchedDomain2D(DIMS, REFS, zRay, HofX, dHdX)

#%% Make the plot
m2k = 1.0E-3
plt.figure(figsize=(12.0, 6.0), tight_layout=True)
plt.subplot(1,2,1)
plt.plot(m2k * XL2[0,:], m2k * ZTL2[0:-1:4,:].T, 'k')
plt.fill_between(m2k * XL2[0,:], m2k * ZTL2[0,:], color='black')
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 12.0)
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
plt.subplot(1,2,2)
plt.plot(m2k * XL1[0,:], m2k * ZTL1[0:-1:4,:].T, 'k')
plt.fill_between(m2k * XL1[0,:], m2k * ZTL1[0,:], color='black')
plt.xlim(-20.0, 20.0)
plt.ylim(0.0, 12.0)
plt.xlabel('X (km)')
plt.savefig('Guerra2020Figure1.pdf')
plt.show()