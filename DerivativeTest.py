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

def function0(x):
       
       Y = np.cos(mt.pi * x)
       DY = -mt.pi * np.sin(mt.pi * x)
       
       return Y, DY

def function1(x):
       
       x2 = np.multiply(x, x)
       Y = 5.0 * np.exp(-5.0 * x) + \
       np.cos(4.0 * mt.pi * x2)
       
       DY = -25.0 * np.exp(-5.0 * x)
       term1 = 8.0 * mt.pi * x
       term2 = np.sin(4.0 * mt.pi * x2)
       DY -= np.multiply(term1, term2)

       return Y, DY

def function2(x):
       
       A = 5.0
       B = 3.0
       C = 4.0
       Y = C * np.exp(-A / ZH * x) * np.sin(B * mt.pi / ZH * x)
       DY = -(A * C) / ZH * np.exp(-A / ZH * x) * np.sin(B * mt.pi / ZH * x)
       DY += (B * C) * mt.pi / ZH * np.exp(-A / ZH * x) * np.cos(B * mt.pi / ZH * x)
       
       return Y, DY

L2 = 1.0E4 * 3.0
L1 = -L2
ZH = 42000.0

#%% SCSE DIFF DERIVATIVE TEST
NX = 30
NZ = 16
NE = 6
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS, False, False, False, True, False)

# NEW Spectral Cubic Spline Derivative matrix
DDZ_SCS, ZE = derv.computeSCSElementDerivativeMatrix(REFS[0], NE)
DDZ2_SCS = DDZ_SCS.dot(DDZ_SCS)
DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(ZE, False, True, False, False, None)

zv = (1.0 / ZH) * ZE
Y1, DY1 = function1(zv)

DYD1 = ZH * DDZ_SCS.dot(Y1)
DYD2 = ZH**2 * DDZ2_SCS.dot(Y1)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y1, label='Function')
plt.plot(zv, DY1, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD1, 'ko--', label='SCSE 1st Derivative')
plt.plot(zv, DYD2, 'go--', label='SCSE 2nd Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Spectral Cubic Spline Elements')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_SCSE.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, (DY1 - DYD1), 'ko-', label='SCSE 1st Derivative')
plt.xlabel('Domain')
plt.ylabel('Derivative Error')
plt.title('Spectral Cubic Spline Elements')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestZ_SCSE.png")

#%% SPECTRAL TRANSFORM TESTS

# Set grid dimensions and order
NX = 128
NZ = 64
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, True, False, False)

#% Compute the raw derivative matrix operators in alpha-xi computational space
DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
#DDX_1D, HF_TRANS, dummy = derv.computeFourierDerivativeMatrix(DIMS)
DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
DDZ2C = DDZ_1D.dot(DDZ_1D)

#DDX_CFD = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[0])
DDX_CFD, DDX2A_CFD = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, False, False, DDX_1D)

#DDZ_CFD = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[1])
DDZ_CFD, DDZ2A_CFD = derv.computeCubicSplineDerivativeMatrix(REFS[1], True, False, False, False,DDZ_1D)
DDZ2A_CFD = DDZ_CFD.dot(DDZ_CFD)
DDZ2B_CFD = derv.computeCompactFiniteDiffDerivativeMatrix2(DIMS, REFS[1])

#%% COMPACT FINITE DIFF DERIVATIVE TEST
zv = (1.0 / ZH) * REFS[1]
Y, DY = function1(zv)

DYD1 = ZH * DDZ_CFD.dot(Y)
DYD2 = ZH * DDZ_1D.dot(Y)
DYD2_1 = ZH**2 * DDZ2A_CFD.dot(Y)
DYD2_2 = ZH**2 * DDZ2B_CFD.dot(Y)
DYD2_3 = ZH**2 * DDZ2C.dot(Y)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD1, 'ro', label='Cubic Spline Derivative')
plt.plot(zv, DYD2, 'r+', label='Chebyshev Derivative')
plt.plot(zv, DYD2_1, 'go', label='Cubic Spline 2nd Derivative')
plt.plot(zv, DYD2_2, 'g+', label='Compact FD 2nd Derivative')
plt.plot(zv, DYD2_3, 'g-', label='Chebyshev 2nd Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('4th Order CFD Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_CFD.png")

#%% HERMITE FUNCTION DERIVATIVE TEST
xv = REFS[0]
# Set the terrain options0
h0 = 2000.0
aC = 5000.0
lC = 4000.0
kC = 1.5E+4
HOPT = [h0, aC, lC, kC, False, 1]

HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT, DDX_1D)
DYD_SPT = DDX_1D.dot(HofX)    
DYD_CFD = DDX_CFD.dot(HofX)

#NX = 16
NZ = 25
NE = 5
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, False, True, False)
DDX_SCS, xe = derv.computeSCSElementDerivativeMatrix(REFS[0], NE)
DDX_CSN, temp = derv.computeCubicSplineDerivativeMatrix(xe, False, True, False, False, None)
REFS[0] = xe
HofX1, dHdX1 = top.computeTopographyOnGrid(REFS, HOPT, DDX_SCS)
DYD_CSN = DDX_CSN.dot(HofX1)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xv, DYD_SPT, 'r-', label='Spectral Derivative')
plt.plot(xv, DYD_CFD, 'bs--', label='Cubic Spline Derivative')
plt.plot(xe, dHdX1, 'go--', label='Spectral CS Element Derivative')
plt.plot(xe, DYD_CSN, 'k+--', label='Natural Cubic Spline Derivative')
#plt.xlim(-12500.0, 12500.0)
plt.xlabel('Domain')
plt.ylabel('Slope')
plt.title('Hermite Function Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestX.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xv, (DYD_SPT - DYD_CFD), 'ko-', label='Hermite Sample')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestX.png")

#%% LAGUERRE FUNCTION DERIVATIVE TEST
NZ = 64
DIMS[-1] = NZ
import HerfunChebNodesWeights as hcl

xi, whf = hcl.lgfunclb(NZ) #[0 inf]
zv1 = 1.0 / np.amax(xi) * xi
Y1, DY1 = function1(zv1)
DDZ_LG, LG_TRANS = derv.computeLaguerreDerivativeMatrix(DIMS)
DYD_LG = ZH * DDZ_LG.dot(Y1)

xi, whf = hcl.leglb(NZ) #[-1 1]
zv2 = (0.5 * (xi + 1.0))
Y2, DY2 = function1(zv2)
DDZ_LD, LD_TRANS = derv.computeLegendreDerivativeMatrix(DIMS)
DYD_LD = ZH * DDZ_LD.dot(Y2)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv2, Y2, label='Function')
plt.plot(zv2, DY2, 'rs-', label='Analytical Derivative')
plt.plot(zv1, DYD_LG, 'k--', label='Laguerre Derivative')
plt.plot(zv2, DYD_LD, 'b--', label='Legendre Derivative')
plt.ylim([1.5*min(DY2), 1.5*max(DY2)])
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Spectral Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_Laguerre-Legendre.png")

#%% Report eigenvalues
plt.show()
import scipy.sparse.linalg as spl
DDZM = DDZ_LG.astype(dtype=np.double)
DZ_eig = spl.eigs(DDZM, k=8, which='LM', return_eigenvectors=False)
print('Laguerre vertical derivative eigenvalues', DZ_eig)
DDZM = DDZ_1D.astype(dtype=np.double)
DZ_eig = spl.eigs(DDZM, k=8, which='LM', return_eigenvectors=False)
print('Chebyshev vertical derivative eigenvalues', DZ_eig)
DDZM = DDZ_LG.astype(dtype=np.double)
DZ_eig = spl.eigs(DDZM, k=8, which='LM', return_eigenvectors=False)
print('Legendre vertical derivative eigenvalues', DZ_eig)
