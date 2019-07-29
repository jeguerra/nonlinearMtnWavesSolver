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

import time
import numpy as np
import math as mt
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import scipy.linalg as sln
from matplotlib import cm
import matplotlib.pyplot as plt
# Import from the local library of routines
from computeGrid import computeGrid
from computeAdjust4CBC import computeAdjust4CBC
from computeColumnInterp import computeColumnInterp
from computeHermiteFunctionDerivativeMatrix import computeHermiteFunctionDerivativeMatrix
from computeChebyshevDerivativeMatrix import computeChebyshevDerivativeMatrix
from computeTopographyOnGrid import computeTopographyOnGrid
from computeGuellrichDomain2D import computeGuellrichDomain2D
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeEulerEquationsLogPLogT import computeEulerEquationsLogPLogT
from computeRayleighEquations import computeRayleighEquations

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
       NZ = 85
       numVar = 4
       iU = 0
       iW = 1
       iP = 2
       iT = 3
       varDex = [iU, iW, iP, iT]
       DIMS = [L1, L2, ZH, NX, NZ]
       OPS = NX * NZ
       
       # Set the terrain options
       h0 = 10.0
       aC = 5000.0
       lC = 4000.0
       HOPT = [h0, aC, lC]
       
       # Set the Rayleigh options
       depth = 10000.0
       width = 20000.0
       applyTop = True
       applyLateral = True
       mu = [1.0E-2, 1.0E-2, 1.0E-2, 1.0E-2]
       
       # Define the computational and physical grids+
       REFS = computeGrid(DIMS)
       
       #%% Compute the raw derivative matrix operators in alpha-xi computational space
       DDX_1D, HF_TRANS = computeHermiteFunctionDerivativeMatrix(DIMS)
       DDZ_1D, CH_TRANS = computeChebyshevDerivativeMatrix(DIMS)
       
       # Update the REFS collection
       REFS.append(DDX_1D)
       REFS.append(DDZ_1D)
       
       #%% Read in topography profile or compute from analytical function
       AGNESI = 1 # "Witch of Agnesi" profile
       SCHAR = 2 # Schar mountain profile nominal (Schar, 2001)
       EXPCOS = 3 # Even exponential and squared cosines product
       EXPPOL = 4 # Even exponential and even polynomial product
       INFILE = 5 # Data from a file (equally spaced points)
       HofX, dHdX = computeTopographyOnGrid(REFS, SCHAR, HOPT)
       
       # Compute the terrain derivatives by Hermite-Function derivative matrix
       dHdX = DDX_1D.dot(HofX)
       #plt.figure()
       #plt.plot(REFS[0], dHdX)
       
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
       
       # Read in or compute background horizontal wind profile
       MEANJET = 1 # Analytical smooth jet profile
       JETOPS = [10.0, 16.822, 1.386]

       U, dUdz = computeShearProfileOnGrid(REFS, JETOPS, P0, PZ, dlnPdz)
       
       #%% Compute the background gradients in physical 2D space
       dUdz = np.expand_dims(dUdz, axis=1)
       DUDZ = np.tile(dUdz, NX)
       DUDZ = computeColumnInterp(DIMS, REFS[1], dUdz, 0, ZTL, DUDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       dlnPdz = np.expand_dims(dlnPdz, axis=1)
       DLPDZ = np.tile(dlnPdz, NX)
       DLPDZ = computeColumnInterp(DIMS, REFS[1], dlnPdz, 0, ZTL, DLPDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       dlnPTdz = np.expand_dims(dlnPTdz, axis=1)
       DLPTDZ = np.tile(dlnPTdz, NX)
       DLPTDZ = computeColumnInterp(DIMS, REFS[1], dlnPTdz, 0, ZTL, DLPTDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       POR = np.expand_dims(POR, axis=1)
       PORZ = np.tile(POR, NX)
       PORZ = computeColumnInterp(DIMS, REFS[1], POR, 0, ZTL, PORZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       U = np.expand_dims(U, axis=1)
       UZ = np.tile(U, NX)
       UZ = computeColumnInterp(DIMS, REFS[1], U, 0, ZTL, UZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       
       # Update the REFS collection
       REFS.append(UZ)
       REFS.append(PORZ)
       REFS.append(DUDZ)
       REFS.append(DLPDZ)
       REFS.append(DLPTDZ)
       
       # Get some memory back here
       del(PORZ)
       del(DUDZ)
       del(DLPDZ)
       del(DLPTDZ)
       
       #%% Get the 2D operators...
       DOPS = computeEulerEquationsLogPLogT(DIMS, PHYS, REFS)
       ROPS = computeRayleighEquations(DIMS, REFS, mu, depth, width, applyTop, applyLateral)
       
       #%% Compute the BC index vector
       ubdex, wbdex, sysDex = computeAdjust4CBC(DIMS, numVar, varDex)
       
       #%% Initialize the global solution vector
       SOL = np.zeros((NX * NZ,1))
       
       #%% Compute the global LHS operator (with Rayleigh terms)
       # Format is 'lil' to allow for column adjustments to the operator
       LDG = sps.bmat([[np.add(DOPS[0], ROPS[0]), DOPS[1], DOPS[2], None], \
                       [None, np.add(DOPS[3], ROPS[1]), DOPS[4], DOPS[5]], \
                       [DOPS[6], DOPS[7], np.add(DOPS[8], ROPS[2]), None], \
                       [None, DOPS[9], None, np.add(DOPS[10], ROPS[3])]], format='lil')
       
       # Get some memory back
       del(DOPS)
       del(ROPS)
       print('Compute global LHS sparse operator: DONE!')
       
       #%% Apply the coupled multipoint constraint for terrain
       DHDXM = sps.spdiags(DZT[0,:], 0, NX, NX)
       # Compute LHS column adjustment to LDG
       LDG[:,ubdex] = np.add(LDG[:,ubdex], (LDG[:,wbdex]).dot(DHDXM))
       # Compute RHS adjustment to forcing
       WBC = np.multiply(DZT[0,:], UZ[0,:])
       # Get some memory back
       del(DHDXM)
       print('Apply coupled BC adjustments: DONE!')
       
       #%% Set up the global solve
       A = LDG[np.ix_(sysDex,sysDex)]
       b = -(LDG[:,wbdex]).dot(WBC)
       del(LDG)
       del(WBC)
       print('Set up global system: DONE!')
       
       #%% Compute the normal equations NO GOOD IN PYTHON! CHOLMOD UNAVAILABLE
       '''
       AN = (A.T).dot(A)
       bN = (A.T).dot(b[sysDex])
       del(A)
       del(b)
       print('Compute the normal equations: DONE!')
       '''
       #%% Solve the system
       start = time.time()
       # Direct solver
       decomp = spl.splu(A.tocsc())
       sol = decomp.solve(b[sysDex])
       #decomp = spl.spilu(A.tocsc())
       #sol0 = decomp.solve(b[sysDex])
       #sol = spl.lgmres(A.tocsc(), b[sysDex], x0=None, atol=1.0E-6, maxiter=1000, inner_m=20)
       endt = time.time()
       print('Solve the system: DONE!')
       print('Elapsed time: ', endt - start)
       
       #%% Recover the solution
       SOL = np.zeros(numVar * NX*NZ)
       SOL[sysDex] = sol[0];
       SOL[wbdex] = np.multiply(DZT[0,:], np.add(UZ[0,:], SOL[ubdex]));
       
       #%% Get the fields in physical space
       udex = np.array(range(OPS))
       wdex = np.add(udex, iW * OPS)
       pdex = np.add(udex, iP * OPS)
       tdex = np.add(udex, iT * OPS)
       uxz = np.reshape(SOL[udex], (NZ, NX), order='F');
       wxz = np.reshape(SOL[wdex], (NZ, NX), order='F');
       pxz = np.reshape(SOL[pdex], (NZ, NX), order='F');
       txz = np.reshape(SOL[tdex], (NZ, NX), order='F');
       print('Recover solution on native grid: DONE!')
       
       #%% Interpolate columns to a finer grid for plotting
       NXI = 3000
       NZI = 500
       uxzint, ZTLI = computeColumnInterp(DIMS, None, None, NZI, ZTL, uxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       wxzint, ZTLI = computeColumnInterp(DIMS, None, None, NZI, ZTL, wxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       pxzint, ZTLI = computeColumnInterp(DIMS, None, None, NZI, ZTL, pxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       txzint, ZTLI = computeColumnInterp(DIMS, None, None, NZI, ZTL, txz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       print('Interpolate columns to finer grid: DONE!')
       
       #%%''' #Spot check the solution on both grids
       fig = plt.figure()
       ccheck = plt.contourf(XL, ZTL, wxz, 101, cmap=cm.seismic)
       cbar = fig.colorbar(ccheck)
       #
       fig = plt.figure()
       ccheck = plt.contourf(wxzint, 101, cmap=cm.seismic)
       cbar = fig.colorbar(ccheck)