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
       
       A = 5.0
       B = 8.0
       C = 5.0
       
       x2 = np.multiply(x, x)
       
       Y = A * np.exp(-C * x) + \
       np.cos(B * mt.pi * x2)
       
       DY = -(A*C) * np.exp(-C * x)
       term1 = (2*B) * mt.pi * x
       term2 = np.sin(B * mt.pi * x2)
       DY -= np.multiply(term1, term2)

       return Y, DY

def function2(x, L):
       
       A = 4.0
       B = 2.0
       C = 4.0
       Y = C * np.exp(-A / L * x) * np.sin(B * mt.pi / L * x)
       DY = -(A * C) / L * np.exp(-A / L * x) * np.sin(B * mt.pi / L * x)
       DY += (B * C) * mt.pi / L * np.exp(-A / L * x) * np.cos(B * mt.pi / L * x)
       
       return Y, DY

L2 = 1.0E4 * 3.0
L1 = -L2
ZH = 42000.0

NZ = 2
NNX = 20
NNZ = 20
NEX = 5
NEZ = 5

NX = NNX * NEX

DIMS = [L1, L2, ZH, NX, NZ]
DIMS_EL = [L1, L2, ZH, NNZ, NZ]
# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, False, True, False)
REFS_EL = computeGrid(DIMS_EL, True, False, False, True, False)

#%% SCSE VS SPECTRAL TEST Z DIRECTION
'''
# NEW Spectral Cubic Spline Derivative matrix
#DDZ_SEM1, ZE1 = derv.computeSpectralElementDerivativeMatrix5E(REFS_EL[0], NEZ, True, 10)
#DDZ_SEM2, ZE2 = derv.computeSpectralElementDerivativeMatrix5E(REFS_EL[0], NEZ, False, 10)
DDZ_SEM1, ZE1 = derv.computeSpectralElementDerivativeMatrix(REFS_EL[0], NEZ, True, (False, False), 10)
DDZ_SEM3 = derv.computeCompactFiniteDiffDerivativeMatrix1(ZE1, 6)
DDZ_SEM4 = derv.computeCompactFiniteDiffDerivativeMatrix1(ZE1, 10)

DL = abs(L2)
zv1 = (1.0 / DL) * ZE1
Y1, DY1 = function1(zv1)

DYD1_SEM1 = DL * DDZ_SEM1.dot(Y1)
DYD1_SEM3 = DL * DDZ_SEM3.dot(Y1)
DYD1_SEM4 = DL * DDZ_SEM4.dot(Y1)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv1, Y1, label='Function')
plt.plot(zv1, DY1, 'r-', label='Analytical Derivative')
plt.plot(zv1, DYD1_SEM1, 'gs--', label='SEM 1st Derivative')
plt.plot(zv1, DYD1_SEM3, 'md--', label='CFD 6 1st Derivative')
plt.plot(zv1, DYD1_SEM4, 'rp--', label='CFD 10 1st Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Spectral Elements')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_SCSE.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv1, np.abs(DYD1_SEM1 - DY1), 'gs-', label='SEM 1 |Error|')
plt.plot(zv1, np.abs(DYD1_SEM3 - DY1), 'md--', label='CFD 6 |Error|')
plt.plot(zv1, np.abs(DYD1_SEM4 - DY1), 'rd--', label='CFD 10 |Error|')
plt.xlabel('Domain')
plt.ylabel('Derivative Error')
plt.title('Spectral Element Coupling Error')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestZ_SCSE.png")
'''
#%% CUBIC and QUINTIC spline derivatives
xv = REFS[0]
# Set the terrain options0
h0 = 2500.0
aC = 10000.0
lC = 7500.0
kC = 2.0E+4
HOPT = [h0, aC, lC, kC, False, 2]

DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)

HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT, DDX_1D)
DYD_SPT = DDX_1D.dot(HofX)    

DDX_CS, DDX2A_CS = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, False, False, DDX_1D)
DDX_QS, DDX4A_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[0], DDX_1D)
DYD_CS = DDX_CS.dot(HofX)
DYD_QS = DDX_QS.dot(HofX)

'''
DDX_SEM1, xe1 = derv.computeSpectralElementDerivativeMatrix(REFS_EL[0], NEX, True, (True, True), 10)
#DDX_SEM1, xe1 = derv.computeSpectralElementDerivativeMatrix5E(REFS_EL[0], NEX, False, 10)
#DDX_SEM2, xe2 = derv.computeSpectralElementDerivativeMatrix5E(REFS_EL[0], NEX, True, 10)
#DDX_SEM2 = derv.computeAdjustedOperatorNBC_ends(DDX_SEM2, DDX_SEM2)
#DDX_SEM2 = derv.computeAdjustedOperatorNBC(DDX_SEM2, DDX_SEM2, -1)

REFS[0] = xe1
HofX1, dHdX1 = top.computeTopographyOnGrid(REFS, HOPT, None)
DYD_SEM1 = DDX_SEM1.dot(HofX1)
'''
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xv, dHdX, 'r-', label='Analytical Derivative')
plt.plot(xv, DYD_SPT, 'r--', label='Spectral Derivative')
plt.plot(xv, DYD_CS, 'bs-', label='Cubic Spline Derivative')
plt.plot(xv, DYD_QS, 'gs-', label='Quintic Derivative')
#plt.xlim(-12500.0, 12500.0)
plt.xlabel('Domain')
plt.ylabel('Slope')
plt.title('Hermite Function Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestX.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xv, np.abs(DYD_SPT - dHdX), 'r--', label='Spectral Derivative')
plt.plot(xv, np.abs(DYD_CS - dHdX), 'gs-', label='CS Derivative Error')
plt.plot(xv, np.abs(DYD_QS - dHdX), 'bs-', label='QS Derivative Error')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestX.png")

print('Spectral Derivative: ', np.count_nonzero(DDX_1D))
print('Cubic Spline Derivative: ', np.count_nonzero(DDX_CS))
print('Quintic Spline Derivative: ', np.count_nonzero(DDX_QS))

#%% SPECTRAL TRANSFORM TESTS

# Set grid dimensions and order
NX = 96
NZ = 96
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, True, False, False)

#% Compute the raw derivative matrix operators in alpha-xi computational space
DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
DDZ2C = DDZ_1D.dot(DDZ_1D)

DDZ_CFD1 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 4)
DDZ_CFD2 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 6)
DDZ_CFD3 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 10)

zv = (1.0 / ZH) * REFS[1]
Y, DY = function1(zv)

DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(zv, True, False, False, False, ZH * DDZ_1D)
DDZ_QS, DDZ4_QS = derv.computeQuinticSplineDerivativeMatrix(zv, ZH * DDZ_1D)

# APPLY A GEOMETRIC MEAN TO THE VARIOUS D MATRICES USING A NORM OF THE SPECTRAL AS REFERENCE

DYD1 = ZH * DDZ_CFD1.dot(Y)
DYD2 = ZH * DDZ_CFD2.dot(Y)
DYD3 = ZH * DDZ_CFD3.dot(Y)
DYD4 = DDZ_CS.dot(Y)
DYD5 = DDZ_QS.dot(Y)
DYD6 = ZH * DDZ_1D.dot(Y)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD1, 'bo--', label='Compact FD4 1st Derivative')
plt.plot(zv, DYD2, 'gp--', label='Compact FD6 1st Derivative')
plt.plot(zv, DYD3, 'kh--', label='Compact FD10 1st Derivative')
plt.plot(zv, DYD4, 'md--', label='Cubic Spline 1st Derivative')
plt.plot(zv, DYD5, 'c+--', label='Quintic Spline 1st Derivative')
plt.plot(zv, DYD6, 'rs--', label='Chebyshev Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Compact Finite Difference Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, np.abs(DYD1 - DY), 'bo--', label='Compact FD4 Error')
plt.plot(zv, np.abs(DYD2 - DY), 'gp--', label='Compact FD6 Error')
plt.plot(zv, np.abs(DYD3 - DY), 'kh--', label='Compact FD10 Error')
plt.plot(zv, np.abs(DYD4 - DY), 'md--', label='Cubic Spline Error')
plt.plot(zv, np.abs(DYD5 - DY), 'c+--', label='Quintic Spline Error')
plt.xlabel('Domain')
plt.ylabel('Error Magnitude')
plt.title('Compact Finite Difference Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_CFD.png")
'''
#%% LAGUERRE FUNCTION DERIVATIVE TEST
NZ = 16
DIMS[-1] = NZ
import HerfunChebNodesWeights as hcl

xi2, whf = hcl.leglb(NZ) #[-1 1]
zv2 = (0.5 * (xi2 + 1.0))
Y2, DY2 = function2(zv2, 1.0)
DDZ_LD, LD_TRANS = derv.computeLegendreDerivativeMatrix(DIMS)
DYD_LD = ZH * DDZ_LD.dot(Y2)

NZC = NZ - 1
DIMS[-1] = NZC
xi3, whf = hcl.cheblb(NZC) #[-1 1]
zv3 = (0.5 * (xi3 + 1.0))
Y3, DY3 = function2(zv3, 1.0)
DDZ_CH, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)

NZL = 16
xi, whf = hcl.lgfunclb(NZL) #[0 inf]
DIMS[-1] = NZL
zv1 = 1.0 / np.amax(xi) * xi
Y1, DY1 = function2(zv1, 1.0)
DDZ_LG, LG_TRANS, scale = derv.computeLaguerreDerivativeMatrix(DIMS)
DYD_LG = ZH * DDZ_LG.dot(Y1)

CTM = hcl.chebpolym(NZC+1, -xi2) # interpolate to legendre grid
LTM, dummy = hcl.legpolym(NZ, xi3, True) # interpolate to chebyshev grid

LG2CH_INT = (LTM.T).dot(LD_TRANS)
CH2LG_INT = (CTM).dot(CH_TRANS)

DDZ_CHS = CH2LG_INT.dot(DDZ_CH).dot(LG2CH_INT)
DDZ_CHS = derv.numericalCleanUp(DDZ_CHS)
DYD_CH = ZH * DDZ_CH.dot(Y3)
DYD_CHS = ZH * DDZ_CHS.dot(Y2)

STG_D = DDZ_LD - DDZ_CHS

plt.figure(figsize=(8, 6), tight_layout=True)
#plt.plot(zv2, Y2, label='Function')
plt.plot(zv2, DY2, 'rs-', label='Analytical Derivative')
plt.plot(zv1, DYD_LG, 'ko--', label='Laguerre Derivative')
plt.plot(zv2, DYD_LD, 'bo--', label='Legendre Derivative')
plt.plot(zv3, DYD_CH, 'go--', label='Chebyshev Derivative')
plt.plot(zv2, DYD_CHS, 'gs--', label='Staggered Chebyshev Derivative')
#plt.ylim([1.5*min(DY2), 1.5*max(DY2)])
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Spectral Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_Laguerre-Legendre.png")

#%% Report eigenvalues
plt.show()
'''
'''
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
'''