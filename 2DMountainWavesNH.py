#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 08:05:02 2019

Computes the transient/static solution to the 2D mountain wave problem.
Log P / Log PT equation set with some static condensation to minimize number of terms.

INPUTS: Piecewise linear T(z) profile sounding (corner points), h(x) topography from
analytical function or equally spaced discrete (FFT interpolation)

COMPUTES: Map of T(z) and h(x) from input to computational domain. Linear LHS operator
matrix, boundary forcing vector and RHS residual. Solves steady problem with UMFPACK and
ALSQR Multigrid. Solves transient problem with Ketchenson SSPRK93 low storage method.

@author: Jorge E. Guerra
"""

import sys
import numpy as np
import math as mt
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from computeGrid import computeGrid
from computeColumnInterp import computeColumnInterp
from computeHermiteFunctionDerivativeMatrix import computeHermiteFunctionDerivativeMatrix
from computeChebyshevDerivativeMatrix import computeChebyshevDerivativeMatrix
from computeTopographyOnGrid import computeTopographyOnGrid
from computeGuellrichDomain2D import computeGuellrichDomain2D
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeEulerEquationsLogPLogT import computeEulerEquationsLogPLogT

if __name__ == '__main__':
       
       # Set physical constants (dry air)
       gc = 9.80601
       P0 = 1.0E5
       cp = 1004.5
       Rd = 287.06
       Kp = Rd / cp
       cv = cp - Rd
       gam = cp / cv
       PHYS = [gc, P0, cp, Rd, Kp, cv, gam]
       
       # Set grid dimensions and order
       L2 = 1.0E4 * 3.0 * mt.pi
       L1 = -L2
       ZH = 36000.0
       NX = 128
       NZ = 96
       DIMS = [L1, L2, ZH, NX, NZ]
       
       # Set the terrain options
       h0 = 10.0
       aC = 5000.0
       lC = 4000.0
       HOPT = [h0, aC, lC]
       
       # Define the computational and physical grids
       REFS = computeGrid(DIMS)
       
       # Compute the raw derivative matrix operators in alpha-xi computational space
       DDX_1D, HF_TRANS = computeHermiteFunctionDerivativeMatrix(DIMS)
       DDZ_1D, CH_TRANS = computeChebyshevDerivativeMatrix(DIMS)
       
       # Update the REFS collection
       REFS.append(DDX_1D)
       REFS.append(DDZ_1D)
       
       ''' #Spot check the derivative matrix
       fig = plt.figure()
       ax = fig.gca(projection='3d')
       X = np.arange(1, NX+1)
       Y = np.arange(1, NX+1)
       X, Y = np.meshgrid(X, Y)
       surf = ax.plot_surface(X, Y, DDX_OP, cmap=cm.coolwarm, linewidth=0)
       '''
       
       #%% Read in topography profile or compute from analytical function
       AGNESI = 1 # "Witch of Agnesi" profile
       SCHAR = 2 # Schar mountain profile nominal (Schar, 2001)
       EXPCOS = 3 # Even exponential and squared cosines product
       EXPPOL = 4 # Even exponential and even polynomial product
       INFILE = 5 # Data from a file (equally spaced points)
       HofX = computeTopographyOnGrid(REFS, SCHAR, HOPT)
       
       # Compute the terrain derivatives...
       dHdX = np.matmul(DDX_1D, HofX.T) 
       
       # Make the 2D physical domains from reference grids and topography
       XL, ZTL, DZT, sigma = computeGuellrichDomain2D(DIMS, REFS, HofX, dHdX)
       # Update the REFS collection
       REFS.append(XL)
       REFS.append(ZTL)
       REFS.append(DZT)
       REFS.append(sigma)
       
       #%% Read in sensible or potential temperature soundings (corner points)
       T_in = [300.0, 228.5, 228.5, 244.5]
       Z_in = [0.0, 1.1E4, 2.0E4, 3.6E4]
       SENSIBLE = 1
       POTENTIAL = 2
       # Map the sounding to the computational vertical grid [0 H]
       TofZ = computeTemperatureProfileOnGrid(Z_in, T_in, REFS)
       # Compute background fields on the vertical
       dlnPdz, LPZ, PZ, dlnPTdz, LPT, PT, RHO = \
              computeThermoMassFields(PHYS, DIMS, REFS, TofZ, SENSIBLE)
              
       # Compute the ratio of pressure to density:
       POR = np.multiply(PZ, np.reciprocal(RHO))
       plt.plot(POR)
       plt.figure()
       
       # Read in or compute background horizontal wind profile
       MEANJET = 1 # Analytical smooth jet profile
       JETOPS = [10.0, 16.822, 1.386]

       U, dUdz = computeShearProfileOnGrid(REFS, JETOPS, P0, PZ, dlnPdz)
       
       #%% Compute the background gradients in physical 2D space with SIGMA
       dUdz = np.expand_dims(dUdz, axis=1)
       DUDZ = np.tile(dUdz, NX)
       DUDZ = np.multiply(sigma, DUDZ)
       dlnPdz = np.expand_dims(dlnPdz, axis=1)
       DLPDZ = np.tile(dlnPdz, NX)
       DLPDZ = np.multiply(sigma, DLPDZ)
       dlnPTdz = np.expand_dims(dlnPTdz, axis=1)
       DLPTDZ = np.tile(dlnPTdz, NX)
       DLPTDZ = np.multiply(sigma, DLPTDZ)
       
       # The following need to be interpolated on a column basis
       POR = np.expand_dims(POR, axis=1)
       PORZ = np.tile(POR, NX)
       PORZ = computeColumnInterp(DIMS, ZTL, PORZ, CH_TRANS)
       U = np.expand_dims(U, axis=1)
       UZ = np.tile(U, NX)
       UZ = computeColumnInterp(DIMS, ZTL, UZ, CH_TRANS)
       
       # Update the REFS collection
       REFS.append(UZ)
       REFS.append(PORZ)
       REFS.append(DUDZ)
       REFS.append(DLPDZ)
       REFS.append(DLPTDZ)
       
       
       #%% Get the 2D operators...
       OPS = NX * NZ
       DOPS = computeEulerEquationsLogPLogT(DIMS, PHYS, REFS)
