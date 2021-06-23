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
ZH = 42000.0
NX = 256
NZ = 128
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, True, False)

#% Compute the raw derivative matrix operators in alpha-xi computational space
DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
#DDX_1D, HF_TRANS, dummy = derv.computeFourierDerivativeMatrix(DIMS)
DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)

#DDX_CFD = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[0])
DDX_CFD, DDX2A_CFD = derv.computeCubicSplineDerivativeMatrix(DIMS, REFS[0], True)

#DDZ_CFD = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[1])
DDZ_CFD, DDZ2A_CFD = derv.computeCubicSplineDerivativeMatrix(DIMS, REFS[1], True)
DDZ2A_CFD = DDZ_CFD.dot(DDZ_CFD)
DDZ2B_CFD = derv.computeCompactFiniteDiffDerivativeMatrix2(DIMS, REFS[1])

#%% COMPACT FINITE DIFF DERIVATIVE TEST
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
plt.plot(zv, DYD, 'k--', label='Cubic Spline Derivative')
plt.plot(zv, DYD2_1, 'g--', label='Cubic Spline 2nd Derivative')
plt.plot(zv, DYD2_2, 'g+', label='Compact FD 2nd Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('4th Order CFD Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestZ_CFD.png")

#%% CHEBYSHEV DERIVATIVE TEST
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

#%% HERMITE FUNCTION DERIVATIVE TEST
xv = REFS[0]
# Set the terrain options0
h0 = 2000.0
aC = 5000.0
lC = 4000.0
kC = 1.25E+4
HOPT = [h0, aC, lC, kC, False, 3]
HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT, DDX_1D)

DYD_SPT = DDX_1D.dot(HofX)    
DYD_CFD = DDX_CFD.dot(HofX)
plt.figure(figsize=(8, 6), tight_layout=True)
#plt.plot(xv, HofX, label='Function')
plt.plot(xv, DYD_SPT, 'r-', label='Spectral Derivative')
plt.plot(xv, DYD_CFD, 'b--', label='Compact CFD Derivative')
#plt.xlim(-12500.0, 12500.0)
plt.xlabel('Domain')
plt.ylabel('Derivatives HF')
plt.title('Hermite Function Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestX.png")

#%% LAGUERRE FUNCTION DERIVATIVE TEST
NZ = 64
DIMS[-1] = NZ
import HerfunChebNodesWeights as hcl
xi, whf = hcl.lgfunclb(NZ) #[0 inf]
zv = ZH / np.amax(xi) * xi

A = 5.0
B = 3.0
C = 4.0
Y = C * np.exp(-A / ZH * zv) * np.sin(B * mt.pi / ZH * zv)
DY = -(A * C) / ZH * np.exp(-A / ZH * zv) * np.sin(B * mt.pi / ZH * zv)
DY += (B * C) * mt.pi / ZH * np.exp(-A / ZH * zv) * np.cos(B * mt.pi / ZH * zv)

#DDZ_CFD, DDZ2A_CFD = derv.computeCubicSplineDerivativeMatrix(DIMS, zv, False)
#DY = DDZ_CFD.dot(Y)

DDZ_LG, LG_TRANS = derv.computeLaguerreDerivativeMatrix(DIMS)
DYD_LG = DDZ_LG.dot(Y)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'rs-', label='Analytical Derivative')
plt.plot(zv, DYD_LG, 'k--', label='Spectral Derivative')
plt.ylim([1.5*min(DY), 1.5*max(DY)])
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Laguerre Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestZ_Laguerre.png")

#%% Report eigenvalues
import scipy.sparse.linalg as spl
DDZM = DDZ_LG.astype(dtype=np.double)
DZ_eig = spl.eigs(DDZM, k=8, which='LM', return_eigenvectors=False)
print('Laguerre vertical derivative eigenvalues', DZ_eig)
DDZM = DDZ_1D.astype(dtype=np.double)
DZ_eig = spl.eigs(DDZM, k=8, which='LM', return_eigenvectors=False)
print('Chebyshev vertical derivative eigenvalues', DZ_eig)

#%% LEGENDRE FUNCTION DERIVATIVE TEST
NZ = 64
DIMS[-1] = NZ
import HerfunChebNodesWeights as hcl
xi, whf = hcl.leglb(NZ) #[0 inf]
zv = ZH * (0.5 * (xi + 1.0))

A = 5.0
B = 3.0
C = 4.0
Y = C * np.exp(-A / ZH * zv) * np.sin(B * mt.pi / ZH * zv)
DY = -(A * C) / ZH * np.exp(-A / ZH * zv) * np.sin(B * mt.pi / ZH * zv)
DY += (B * C) * mt.pi / ZH * np.exp(-A / ZH * zv) * np.cos(B * mt.pi / ZH * zv)

#DDZ_CFD, DDZ2A_CFD = derv.computeCubicSplineDerivativeMatrix(DIMS, zv, False)
#DY = DDZ_CFD.dot(Y)

DDZ_LG, LG_TRANS = derv.computeLegendreDerivativeMatrix(DIMS)
DYD_LG = DDZ_LG.dot(Y)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'rs-', label='Analytical Derivative')
plt.plot(zv, DYD_LG, 'k--', label='Spectral Derivative')
plt.ylim([1.5*min(DY), 1.5*max(DY)])
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Legendre Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
plt.savefig("DerivativeTestZ_Legendre.png")

#%% Report eigenvalues
import scipy.sparse.linalg as spl
DDZM = DDZ_LG.astype(dtype=np.double)
DZ_eig = spl.eigs(DDZM, k=8, which='LM', return_eigenvectors=False)
print('Legendre vertical derivative eigenvalues', DZ_eig)
