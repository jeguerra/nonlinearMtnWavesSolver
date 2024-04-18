#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:24:39 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
import scipy.linalg as scl
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
       
       x2 = x*x
       
       Y = A * np.exp(-C * x) + \
       np.cos(B * mt.pi * x2)
       
       DY = -(A*C) * np.exp(-C * x)
       term1 = (2*B) * mt.pi * x
       term2 = np.sin(B * mt.pi * x2)
       DY -= np.multiply(term1, term2)
       
       for ii in range(len(x)):
              if x[ii] >= 0.5:
                     Y[ii] = 1.0 - 2.0*x[ii]
                     DY[ii] = -2.0

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
REFS = computeGrid(DIMS, None, True, False, False, True)
REFS_EL = computeGrid(DIMS_EL, None, True, False, False, True)

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
plt.grid(visible=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_SCSE.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv1, np.abs(DYD1_SEM1 - DY1), 'gs-', label='SEM 1 |Error|')
plt.plot(zv1, np.abs(DYD1_SEM3 - DY1), 'md--', label='CFD 6 |Error|')
plt.plot(zv1, np.abs(DYD1_SEM4 - DY1), 'rd--', label='CFD 10 |Error|')
plt.xlabel('Domain')
plt.ylabel('Derivative Error')
plt.title('Spectral Element Coupling Error')
plt.grid(visible=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestZ_SCSE.png")
'''
#%% CUBIC and QUINTIC spline derivatives
xv = REFS[0]
# Set the terrain options0
h0 = 2500.0
aC = 5000.0
lC = 2.0 * mt.pi * 1.0E3
kC = 1.5E+4
HOPT = [h0, aC, lC, kC, False, 2]

DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)

HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT)
DYD_SPT = DDX_1D.dot(HofX)    

#DDX_BC = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 4)
DDX_CS, DDX2A_CS = derv.computeCubicSplineDerivativeMatrix(REFS[0], False, True, 0.0)
#DDX_BC = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 6)
DDX_QS, DDX4A_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[0], False, True, DDX_CS)

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
plt.grid(visible=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestX.png")

plt.figure(figsize=(8, 6), tight_layout=True)
plt.semilogy(xv, np.abs(DYD_SPT - dHdX), 'r--', label='Spectral Derivative')
plt.semilogy(xv, np.abs(DYD_CS - dHdX), 'bs-', label='CS Derivative Error')
plt.semilogy(xv, np.abs(DYD_QS - dHdX), 'gs-', label='QS Derivative Error')
plt.grid(visible=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeErrorTestX.png")

plt.show()
print('Spectral Derivative: ', np.count_nonzero(DDX_1D))
print('Cubic Spline Derivative: ', np.count_nonzero(DDX_CS))
print('Quintic Spline Derivative: ', np.count_nonzero(DDX_QS))

#%% SPECTRAL TRANSFORM TESTS

# Set grid dimensions and order
NX = 96
NZ = 95
DIMS = [L1, L2, ZH, NX, NZ]

# Define the computational and physical grids+
REFS_CH = computeGrid(DIMS, None, True, False, True, False)
REFS_LG = computeGrid(DIMS, None, True, False, False, True)

#% Compute the raw derivative matrix operators in alpha-xi computational space
DDZ_LG, LG_TRANS = derv.computeLegendreDerivativeMatrix(DIMS)
DDZ2L = DDZ_LG.dot(DDZ_LG)
DDZ_CH, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
DDZ2C = DDZ_CH.dot(DDZ_CH)

zv = (1.0 / ZH) * REFS_CH[1]
Y, DY = function1(zv)

zvl = (1.0 / ZH) * REFS_LG[1]
YL, DYL = function1(zvl)

DDZ_CFD1 = derv.computeCompactFiniteDiffDerivativeMatrix1(zv, 6)
DDZ_CFD2 = derv.computeCompactFiniteDiffDerivativeMatrix1(zv, 8)
DDZ_CFD3 = derv.computeCompactFiniteDiffDerivativeMatrix1(zv, 10)
print('Compact FD4: ', np.count_nonzero(DDZ_CFD1))
print('Compact FD6: ', np.count_nonzero(DDZ_CFD2))
print('Compact FD8: ', np.count_nonzero(DDZ_CFD3))

# Take boundary information from compact FD operators
DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(zv, False, True, None)
DDZ_RS, DDZ3_RS = derv.computeQuarticSplineDerivativeMatrix(zv, False, True, DDZ_CS)
DDZ_QS, DDZ4_QS = derv.computeQuinticSplineDerivativeMatrix(zv, False, True, DDZ_CS)

# Compute eigenspectra
W1 = scl.eigvals(DDZ_CFD2)
W2 = scl.eigvals(DDZ_CFD3)
W3 = scl.eigvals(DDZ_CS)
W4 = scl.eigvals(DDZ_RS)
W5 = scl.eigvals(DDZ_QS)
W6 = scl.eigvals(ZH * DDZ_LG)
W7 = scl.eigvals(ZH * DDZ_CH)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(np.real(W1), np.imag(W1), 'o', label='Compact FD8')
plt.plot(np.real(W2), np.imag(W2), 'o', label='Compact FD10')
plt.plot(np.real(W3), np.imag(W3), 'o', label='Cubic Spline') 
plt.plot(np.real(W4), np.imag(W4), 'o', label='Quartic Spline')
plt.plot(np.real(W5), np.imag(W5), 'o', label='Quintic Spline')
plt.plot(np.real(W6), np.imag(W6), 'o', label='Legendre') 
plt.plot(np.real(W7), np.imag(W7), 'o', label='Chebyshev')
plt.grid(visible=True, which='both', axis='both')
plt.legend()

'''
# Compute SVD of derivative matrices
U1, s1, Vh1 = scl.svd(DDX_1D)
U2, s2, Vh2 = scl.svd(DDZ_1D)
U3, s3, Vh3 = scl.svd(DDZ_CFD1)
U4, s4, Vh4 = scl.svd(DDZ_CFD2)
U5, s5, Vh5 = scl.svd(DDZ_CFD3)
U6, s6, Vh6 = scl.svd(DDZ_CS)
U7, s7, Vh7 = scl.svd(DDZ_QS)

plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(s1 / s1[0], 'o', label='Hermite Function'); plt.plot(s2 / s2[0], 'o', label='Chebychev') 
plt.plot(s3 / s3[0], 'o', label='Compact FD4'); plt.plot(s4 / s4[0], 'o', label='Compact FD6') 
plt.plot(s5 / s5[0], 'o', label='Compact FD10'); plt.plot(s6 / s6[0], 'o', label='Cubic Spline') 
plt.plot(s7 / s7[0], 'o', label='Quintic Spline')
plt.grid(visible=True, which='both', axis='both')
plt.legend()
'''
DYD1 = DDZ_CFD1.dot(Y)
DYD2 = DDZ_CFD2.dot(Y)
DYD3 = DDZ_CFD3.dot(Y)
DYD4 = DDZ_CS.dot(Y)
DYD5 = DDZ_RS.dot(Y)
DYD6 = DDZ_QS.dot(Y)
DYD7 = ZH * DDZ_CH.dot(Y)
DYD8 = ZH * DDZ_LG.dot(YL)
plt.figure(figsize=(8, 6), tight_layout=True)
plt.plot(zv, Y, label='Function')
plt.plot(zv, DY, 'r-', label='Analytical Derivative')
plt.plot(zv, DYD1, 'bo--', label='Compact FD6 1st Derivative')
plt.plot(zv, DYD2, 'gp--', label='Compact FD8 1st Derivative')
plt.plot(zv, DYD3, 'kh--', label='Compact FD10 1st Derivative')
plt.plot(zv, DYD4, 'md--', label='Cubic Spline 1st Derivative')
plt.plot(zv, DYD5, '+--', label='Quartic Spline 1st Derivative', color='orange')
plt.plot(zv, DYD6, 'c+--', label='Quintic Spline 1st Derivative')
plt.plot(zv, DYD7, 'rs--', label='Chebyshev Derivative')
plt.plot(zvl, DYD8, 'ks--', label='Legendre Derivative')
plt.xlabel('Domain')
plt.ylabel('Functions')
plt.title('Compact Finite Difference Derivative Test')
plt.grid(visible=True, which='both', axis='both')
plt.legend()

plt.figure(figsize=(8, 6), tight_layout=True)
plt.semilogy(zv, np.abs(DYD1 - DY), 'bo--', label='Compact FD6 Error')
plt.semilogy(zv, np.abs(DYD2 - DY), 'gp--', label='Compact FD8 Error')
plt.semilogy(zv, np.abs(DYD3 - DY), 'kh--', label='Compact FD10 Error')
plt.semilogy(zv, np.abs(DYD4 - DY), 'md--', label='Cubic Spline Error', linewidth=2.0)
plt.semilogy(zv, np.abs(DYD5 - DY), 'o--', label='Quartic Spline Error', linewidth=2.0, color='orange')
plt.semilogy(zv, np.abs(DYD6 - DY), 'c+--', label='Quintic Spline Error', linewidth=2.0)
plt.semilogy(zv, np.abs(DYD7 - DY), 'rs--', label='Chebyshev Spectral Error')
plt.semilogy(zv, np.abs(DYD8 - DYL), 'ks--', label='Legendre Spectral Error')
plt.xlabel('Domain')
plt.ylabel('Error Magnitude')
plt.title('Compact Finite Difference Derivative Test')
plt.grid(visible=True, which='both', axis='both')
plt.legend()
#plt.savefig("DerivativeTestZ_CFD.png")

plt.show()
