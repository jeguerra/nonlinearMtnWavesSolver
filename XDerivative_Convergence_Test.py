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
NX = [127, 255, 511]
NZ = 128

for ii in range(len(NX)):
       DIMS = [L1, L2, ZH, NX[ii], NZ]
       
       # Define the computational and physical grids+
       REFS = computeGrid(DIMS, True, False, True)
       
       #% Compute the raw derivative matrix operators in alpha-xi computational space
       DDX_1DS, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       DDX_1DS[np.isclose(DDX_1DS, 0.0, atol=1.0E-4)] = 0.0
       DDX_CFD = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[0])
       DDX_CSD = derv.computeCubicSplineDerivativeMatrix(DIMS, REFS[0], True)
       
       #DDX_1DS_AVG = 0.5 * (DDX_1DS + DDX_CFD)
       
       # HERMITE FUNCTION DERIVATIVE TEST
       xv = REFS[0]
       
       # Set the terrain options0
       h0 = 2000.0
       aC = 5000.0
       lC = 4000.0
       kC = 1.25E+4
       HOPT = [h0, aC, lC, kC, False, 3]
       #HofX, dHdX = top.computeTopographyOnGrid(REFS, HOPT, DDX_1DS)
       HofX = np.ones(NX[ii]+1)
           
       dHdX_SPD = DDX_1DS.dot(HofX)
       dHdX_CFD = DDX_CFD.dot(HofX)
       dHdX_CSD = DDX_CSD.dot(HofX)
       
       plt.figure(figsize=(8, 6), tight_layout=True)
       plt.plot(xv, dHdX_SPD, 'r-', label='Spectral Derivative')
       plt.plot(xv, dHdX_CFD, 'b--', label='Compact CFD Derivative')
       plt.plot(xv, dHdX_CSD, 'g--', label='Cubic Spline Derivative')
       #plt.xlim(-15000.0, 15000.0)
       plt.xlabel('X')
       plt.ylabel('Derivative')
       plt.title('Derivative Comparison')
       plt.grid(b=True, which='both', axis='both')
       plt.legend()
       plt.savefig("DerivativeComparisonX.png")
