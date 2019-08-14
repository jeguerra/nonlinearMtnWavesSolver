#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:24:39 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt
from computeGrid import computeGrid
import computeHermiteFunctionDerivativeMatrix as hfd
import computeChebyshevDerivativeMatrix as chd

#%%
# Make a test function and its derivative (DEBUG)

 # Set grid dimensions and order
L2 = 1.0E4 * 3.0 * mt.pi
L1 = -L2
ZH = 36000.0
NX = 64
NZ = 64
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS)

#% Compute the raw derivative matrix operators in alpha-xi computational space
DDX_1D, HF_TRANS = hfd.computeHermiteFunctionDerivativeMatrix(DIMS)
DDZ_1D, CH_TRANS = chd.computeChebyshevDerivativeMatrix(DIMS)

# CHEBYSHEV DERIVATIVE TEST
zv = (1.0 / ZH) * REFS[1]
zv2 = np.multiply(zv, zv)
Y = 4.0 * np.exp(-5.0 * zv) + \
np.cos(4.0 * mt.pi * zv2);
DY = -20.0 * np.exp(-5.0 * zv)
term1 = 8.0 * mt.pi * zv
term2 = np.sin(4.0 * mt.pi * zv2)
DY -= np.multiply(term1, term2);
    
DYD = ZH * DDZ_1D.dot(Y)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD, 'k--', label='Spectral Derivative')
plt.xlabel('Domain')
plt.ylabel('Function')
plt.title('Chebyshev Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestZ.png")

# HERMITE FUNCTION DERIVATIVE TEST
xv = (1.0 / L2) * REFS[0]
xv2 = np.multiply(xv, xv)
#Y = 4.0 * np.exp(-5.0 * xv) + \
Y = np.cos(4.0 * mt.pi * xv2);
#DY = -20.0 * np.exp(-5.0 * xv)
term1 = 8.0 * mt.pi * xv
term2 = np.sin(4.0 * mt.pi * xv2)
DY = np.multiply(term1, term2);
    
DYD = -L2 * DDX_1D.dot(Y)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xv, Y, label='Function')
plt.plot(xv, DY, 'r-', label='Analytical Derivative')
plt.plot(xv, DYD, 'k--', label='Spectral Derivative')
plt.xlabel('Domain')
plt.ylabel('Function')
plt.title('Hermite Function Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestX.png")
