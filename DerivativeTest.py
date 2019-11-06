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
import computeDerivativeMatrix as derv
import computeTopographyOnGrid as top

#%%
# Make a test function and its derivative (DEBUG)

 # Set grid dimensions and order
L2 = 1.0E4 * 3.0 * mt.pi
L1 = -L2
ZH = 36000.0
NX = 512
NZ = 128
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS)

#% Compute the raw derivative matrix operators in alpha-xi computational space
DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
DDZ_CFD = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[1])
DDZ2A_CFD = DDZ_CFD.dot(DDZ_CFD)
DDZ2B_CFD = derv.computeCompactFiniteDiffDerivativeMatrix2(DIMS, REFS[1])

# COMPACT FINITE DIFF DERIVATIVE TEST
zv = (1.0 / ZH) * REFS[1]
zv2 = np.multiply(zv, zv)
Y = 4.0 * np.exp(-5.0 * zv) + \
np.cos(4.0 * mt.pi * zv2);
DY = -20.0 * np.exp(-5.0 * zv)
term1 = 8.0 * mt.pi * zv
term2 = np.sin(4.0 * mt.pi * zv2)
DY -= np.multiply(term1, term2);
    
DYD = ZH * DDZ_CFD.dot(Y)
DYD2_1 = ZH**2 * DDZ2A_CFD.dot(Y)
DYD2_2 = ZH**2 * DDZ2B_CFD.dot(Y)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD, 'k--', label='Compact FD Derivative')
plt.plot(zv, DYD2_1, 'g--', label='Compact FD 2nd Derivative 1')
plt.plot(zv, DYD2_2, 'g+', label='Compact FD 2nd Derivative 2')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('4th Order CFD Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestZ_CFD.png")

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
plt.ylabel('Functions')
plt.title('Chebyshev Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestZ.png")

# HERMITE FUNCTION DERIVATIVE TEST
xv = REFS[0]
''' Hermite Functions are NO GOOD for asymmetrical functions in bounded intervals
xv2 = np.multiply(xv, xv)
#Y = 4.0 * np.exp(-5.0 * xv) + \
Y = np.cos(4.0 * mt.pi * xv2);
#DY = -20.0 * np.exp(-5.0 * xv)
term1 = 8.0 * mt.pi * xv
term2 = -np.sin(4.0 * mt.pi * xv2)
DY = np.multiply(term1, term2);
'''
# Set the terrain options0
h0 = 100.0
aC = 5000.0
lC = 4000.0
HOPT = [h0, aC, lC]
HofX, dHdX = top.computeTopographyOnGrid(REFS, 2, HOPT)
    
DYD = DDX_1D.dot(HofX)
plt.figure(figsize=(8, 6), tight_layout=True)
#plt.plot(xv, HofX, label='Function')
plt.plot(xv, dHdX, 'r-', label='Fourier Derivative')
plt.plot(xv, DYD, 'k--', label='Hermite Derivative')
plt.xlim(-12500.0, 12500.0)
plt.xlabel('Domain')
plt.ylabel('Derivatives HF')
plt.title('Hermite Function Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestX.png")
