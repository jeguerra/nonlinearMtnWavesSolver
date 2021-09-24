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
NNZ = 10
NEX = 9
NEZ = 9
NX = NNX * NEX
isAdjusted = True

DIMS = [L1, L2, ZH, NX, NZ]
DIMS_EL = [L1, L2, ZH, NNZ, NZ]
# Define the computational and physical grids+
REFS = computeGrid(DIMS, True, False, False, True, False)
REFS_EL = computeGrid(DIMS_EL, False, True, False, True, False)

#%% SCSE VS SPECTRAL TEST Z DIRECTION

# NEW Spectral Cubic Spline Derivative matrix
DDZ_SCSE, ZE = derv.computeSCSElementDerivativeMatrix(REFS_EL[0], NEZ, False, (1.0,1.0))
DDZ2_SCSE = DDZ_SCSE.dot(DDZ_SCSE)
DDZ_SEM, ZE = derv.computeSpectralElementDerivativeMatrix(REFS_EL[0], NEZ, isAdjusted)
DDZ2_SEM = DDZ_SEM.dot(DDZ_SEM)
#DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(ZE, False, True, False, False, None)

DL = abs(L2)
zv = (1.0 / DL) * ZE
Y1, DY1 = function1(zv)

DYD1_SEM = DL * DDZ_SEM.dot(Y1)
#DYD1_SCSE = DL * DDZ_SCSE.dot(Y1)
DYD2_SEM = DL**2 * DDZ2_SEM.dot(Y1)
#DYD2_SCSE = DL**2 * DDZ2_SCSE.dot(Y1)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y1, label='Function')
plt.plot(zv, DY1, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD1_SEM, 'ko--', label='SEM 1st Derivative')
#plt.plot(zv, DYD1_SCSE, 'ks--', label='SCSE 1st Derivative')
plt.plot(zv, DYD2_SEM, 'g+--', label='SEM 2nd Derivative')
#plt.plot(zv, DYD2_SCSE, 'b+--', label='SCSE 2nd Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Spectral Cubic Spline Elements')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_SCSE.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, np.abs(DYD1_SEM - DY1), 'bo-', label='SEM |Error|')
#plt.plot(zv, np.abs(DYD1_SCSE - DY1), 'gs-', label='SCSE |Error|')
plt.xlabel('Domain')
plt.ylabel('Derivative Error')
plt.title('Spectral Cubic Spline Elements')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestZ_SCSE.png")

#%% SCSE VS SPECTRAL TEST X DIRECTION
xv = REFS[0]
# Set the terrain options0
h0 = 2500.0
aC = 10000.0
lC = 10000.0
kC = 2.5E+4
HOPT = [h0, aC, lC, kC, False, 3]

DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
DDX_SCS, DDX2A_SCS = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, False, False, DDX_1D)

HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT, DDX_1D)
DYD_SPT = DDX_1D.dot(HofX)    
DYD_SCS1 = DDX_SCS.dot(HofX)

#DDX_SCSE, xe = derv.computeSCSElementDerivativeMatrix(REFS_EL[0], NEX, False, (1.0, 1.0))
DDX_SEM, xe = derv.computeSpectralElementDerivativeMatrix(REFS_EL[0], NEX, isAdjusted)
DDX_CSN, temp = derv.computeCubicSplineDerivativeMatrix(xe, False, True, False, False, None)
DDX_SCS, DDX2A_SCS = derv.computeCubicSplineDerivativeMatrix(xe, True, False, False, False, DDX_SEM)
REFS[0] = xe
HofX1, dHdX1 = top.computeTopographyOnGrid(REFS, HOPT, DDX_SEM)
DYD_SEM = DDX_SEM.dot(HofX1)
DYD_CSN = DDX_CSN.dot(HofX1)
DYD_SCS2 = DDX_SCS.dot(HofX1)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xe, dHdX1, 'r-', label='Analytical Derivative')
plt.plot(xv, DYD_SPT, 'r--', label='Spectral Derivative')
#plt.plot(xv, DYD_SCS1, 'bH--', label='Spectral Cubic Spline Derivative')
#plt.plot(xe, DYD_SCS2, 'bD--', label='Spectral Cubic Spline Derivative')
plt.plot(xe, DYD_SEM, 'go--', label='Spectral Element Derivative')
plt.plot(xe, DYD_CSN, 'k+--', label='Natural Cubic Spline Derivative')
#plt.xlim(-12500.0, 12500.0)
plt.xlabel('Domain')
plt.ylabel('Slope')
plt.title('Hermite Function Derivative Test')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestX.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(xv, np.abs(DYD_SPT - dHdX), 'r--', label='Spectral Derivative')
#plt.plot(xv, np.abs(DYD_SCS1 - dHdX), 'bH-', label='Spectral CS Derivative 1')
#plt.plot(xe, np.abs(DYD_SCS2 - dHdX1), 'bD-', label='Spectral CS Derivative 2')
plt.plot(xe, np.abs(DYD_SEM - dHdX1), 'go-', label='SEM Derivative')
plt.plot(xe, np.abs(DYD_CSN - dHdX1), 'k+-', label='CSN Derivative')
plt.grid(b=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestX.png")

print('Spectral Derivative: ', np.count_nonzero(DDX_1D))
print('Spectral CS Derivative: ', np.count_nonzero(DDX_SCS))
print('Natural Cubic Spline Derivative: ', np.count_nonzero(DDX_CSN))
print('Spectral Element Derivative: ', np.count_nonzero(DDX_SEM))

#%% SPECTRAL TRANSFORM TESTS

# Set grid dimensions and order
NX = 48
NZ = 48
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

#%% LAGUERRE FUNCTION DERIVATIVE TEST
NZ = 16
DIMS[-1] = NZ
import HerfunChebNodesWeights as hcl

xi, whf = hcl.lgfunclb(NZ) #[0 inf]
zv1 = 1.0 / np.amax(xi) * xi
Y1, DY1 = function2(zv1, 1.0)
DDZ_LG, LG_TRANS, scale = derv.computeLaguerreDerivativeMatrix(DIMS)
DYD_LG = ZH * DDZ_LG.dot(Y1)

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