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
import time
import shelve
import numpy as np
import math as mt
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from matplotlib import cm
import matplotlib.pyplot as plt
# Import from the local library of routines
from computeGrid import computeGrid
from computeAdjust4CBC import computeAdjust4CBC
from computeColumnInterp import computeColumnInterp
from computeHorizontalInterp import computeHorizontalInterp
from computePartialDerivativesXZ import computePartialDerivativesXZ
from computeHermiteFunctionDerivativeMatrix import computeHermiteFunctionDerivativeMatrix
from computeChebyshevDerivativeMatrix import computeChebyshevDerivativeMatrix
from computeTopographyOnGrid import computeTopographyOnGrid
from computeGuellrichDomain2D import computeGuellrichDomain2D
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeEulerEquationsLogPLogT import computeEulerEquationsLogPLogT
from computeEulerEquationsLogPLogT import computeEulerEquationsLogPLogT_NL
from computeRayleighEquations import computeRayleighEquations
from computeResidualViscCoeffs import computeResidualViscCoeffs
from computeTimeIntegration import computePrepareFields
from computeTimeIntegration import computeTimeIntegrationLN
from computeTimeIntegration import computeTimeIntegrationNL

# Truncated spectral derivative matrices
#import computeHermiteFunctionDerivativeMatrix_Truncated as hfd
#import computeChebyshevDerivativeMatrix_Truncated as chd

if __name__ == '__main__':
       # Set the solution type
       StaticSolve = False
       TransientSolve = False
       NonLinSolve = True
       ResDiff = True
       
       # Set restarting
       toRestart = True
       isRestart = True
       
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
       NX = 129
       NZ = 81
       OPS = (NX + 1) * NZ
       numVar = 4
       iU = 0
       iW = 1
       iP = 2
       iT = 3
       varDex = [iU, iW, iP, iT]
       DIMS = [L1, L2, ZH, NX, NZ, OPS]
       # Make the equation index vectors for all DOF
       udex = np.array(range(OPS))
       wdex = np.add(udex, OPS)
       pdex = np.add(wdex, OPS)
       tdex = np.add(pdex, OPS)
       
       # Set the terrain options
       h0 = 100.0
       aC = 5000.0
       lC = 4000.0
       HOPT = [h0, aC, lC]
       
       # Set the Rayleigh options
       depth = 10000.0
       width = 20000.0
       applyTop = True
       applyLateral = True
       mu = [1.0E-2, 1.0E-2, 1.0E-2, 1.0E-2]
       
       #%% Transient solve parameters
       DT = 0.1 # Linear transient
       #DT = 0.05 # Nonlinear transient
       HR = 0.04
       ET = HR * 60 * 60 # End time in seconds
       OTI = 100 # Stride for diagnostic output
       RTI = 10 # Stride for residual visc update
       
       #%% Define the computational and physical grids+
       REFS = computeGrid(DIMS)
       
       # Compute DX and DZ grid length scales
       DX = np.min(np.diff(REFS[0]))
       DZ = np.min(np.diff(REFS[1]))
       
       #% Compute the raw derivative matrix operators in alpha-xi computational space
       DDX_1D, HF_TRANS = computeHermiteFunctionDerivativeMatrix(DIMS)
       DDZ_1D, CH_TRANS = computeChebyshevDerivativeMatrix(DIMS)
       
       # Update the REFS collection
       REFS.append(DDX_1D)
       REFS.append(DDZ_1D)
       
       #% Read in topography profile or compute from analytical function
       AGNESI = 1 # "Witch of Agnesi" profil e
       SCHAR = 2 # Schar mountain profile nominal (Schar, 2001)
       EXPCOS = 3 # Even exponential and squared cosines product
       EXPPOL = 4 # Even exponential and even polynomial product
       INFILE = 5 # Data from a file (equally spaced points)
       HofX, dHdX = computeTopographyOnGrid(REFS, SCHAR, HOPT)
       
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
       DUDZ = np.tile(dUdz, NX+1)
       DUDZ = computeColumnInterp(DIMS, REFS[1], dUdz, 0, ZTL, DUDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       dlnPdz = np.expand_dims(dlnPdz, axis=1)
       DLPDZ = np.tile(dlnPdz, NX+1)
       DLPDZ = computeColumnInterp(DIMS, REFS[1], dlnPdz, 0, ZTL, DLPDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       dlnPTdz = np.expand_dims(dlnPTdz, axis=1)
       DLPTDZ = np.tile(dlnPTdz, NX+1)
       DLPTDZ = computeColumnInterp(DIMS, REFS[1], dlnPTdz, 0, ZTL, DLPTDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       # Compute the background (initial) fields
       POR = np.expand_dims(POR, axis=1)
       PORZ = np.tile(POR, NX+1)
       PORZ = computeColumnInterp(DIMS, REFS[1], POR, 0, ZTL, PORZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       U = np.expand_dims(U, axis=1)
       UZ = np.tile(U, NX+1)
       UZ = computeColumnInterp(DIMS, REFS[1], U, 0, ZTL, UZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       LPZ = np.expand_dims(LPZ, axis=1)
       LOGP = np.tile(LPZ, NX+1)
       LOGP = computeColumnInterp(DIMS, REFS[1], LPZ, 0, ZTL, LOGP, CH_TRANS, '1DtoTerrainFollowingCheb')
       LPT = np.expand_dims(LPT, axis=1)
       LOGT = np.tile(LPT, NX+1)
       LOGT = computeColumnInterp(DIMS, REFS[1], LPT, 0, ZTL, LOGT, CH_TRANS, '1DtoTerrainFollowingCheb')
       
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
       
       #%% Get the 2D linear operators...
       DDXM, DDZM = computePartialDerivativesXZ(DIMS, REFS, DDX_1D, DDZ_1D)              
       REFS.append(DDXM)
       REFS.append(DDZM)
       REFS.append(DDXM.dot(DDXM))
       REFS.append(DDZM.dot(DDZM))
       DOPS = computeEulerEquationsLogPLogT(DIMS, PHYS, REFS)
       ROPS = computeRayleighEquations(DIMS, REFS, mu, depth, width, applyTop, applyLateral)
       
       #%% Compute the BC index vector
       ubdex, utdex, wbdex, sysDexST, sysDexTR = computeAdjust4CBC(DIMS, numVar, varDex)
       
       if StaticSolve:
              sysDex = sysDexST
       elif TransientSolve or NonLinSolve:
              sysDex = sysDexTR
       
       #%% Initialize the global solution vector
       SOL = np.zeros((NX * NZ,1))
       
       #%% Rayleigh opearator
       RAYOP = sps.block_diag((ROPS[0], ROPS[1], ROPS[2], ROPS[3]), format='lil')
       
       #%% Compute the global LHS operator
       if StaticSolve or TransientSolve:
              # Format is 'lil' to allow for column adjustments to the operator
              LDG = sps.bmat([[DOPS[0], DOPS[1], DOPS[2], None], \
                              [None, DOPS[3], DOPS[4], DOPS[5]], \
                              [DOPS[6], DOPS[7], DOPS[8], None], \
                              [None, DOPS[9], None, DOPS[10]]], format='lil')
              
              # Get some memory back
              del(DOPS)
              print('Compute global sparse linear Euler operator: DONE!')
       
              # Apply the coupled multipoint constraint for terrain
              DHDXM = sps.spdiags(DZT[0,:], 0, NX+1, NX+1)
              # Compute LHS column adjustment to LDG
              LDG[:,ubdex] += (LDG[:,wbdex]).dot(DHDXM)
              # Compute RHS adjustment to forcing
              WBC = DZT[0,:] * UZ[0,:]
              # Get some memory back
              del(DHDXM)
              print('Apply coupled BC adjustments: DONE!')
       
              # Set up the global solve
              A = LDG + RAYOP
              AN = A[np.ix_(sysDex,sysDex)]
              del(A)
              AN = AN.tocsc()
              bN = -(LDG[:,wbdex]).dot(WBC)
              del(LDG)
              del(WBC)
              print('Set up global solution operators: DONE!')
       
       #%% Solve the system - Static or Transient Solution
       start = time.time()
       if StaticSolve:
              sol = spl.spsolve(AN, bN[sysDex], use_umfpack=False)
       elif TransientSolve:
              print('Starting Linear Transient Solver...')
              bN = bN[sysDex]
              
              # Initialize transient storage
              SOLT = np.zeros((numVar * OPS, 2))
              INIT = np.zeros((numVar * OPS,))
              RESI = np.zeros((numVar * OPS,))
              
              # Initialize the Background fields
              INIT[udex] = np.reshape(UZ, (OPS,), order='F')
              INIT[wdex] = np.zeros((OPS,))
              INIT[pdex] = np.reshape(LOGP, (OPS,), order='F')
              INIT[tdex] = np.reshape(LOGT, (OPS,), order='F')
              
              if isRestart:
                     rdb = shelve.open('restartDB', flag='r')
                     SOLT[udex] = rdb['uxz']
                     SOLT[wdex] = rdb['wxz']
                     SOLT[pdex] = rdb['pxz']
                     SOLT[tdex] = rdb['txz']
                     RHS = rdb['RHS']
                     NX_in = rdb['NX']
                     NZ_in = rdb['NZ']
                     IT = rdb['ET']
                     rdb.close()
                     
                     if NX_in != NX or NZ_in != NZ:
                            print('ERROR: RESTART DATA IS INVALID')
                            sys.exit(2)
                            
                     if ET <= IT:
                            print('ERROR: END TIME LEQ INITIAL TIME ON RESTART')
                            sys.exit(2)
                            
                     # Initialize the restart time array
                     TI = np.array(np.arange(IT + DT, ET, DT))
              else:
                     # Initialize time array
                     TI = np.array(np.arange(DT, ET, DT))
                     # Initialize the RHS
                     RHS = bN
                     
              error = [np.linalg.norm(RHS)]
              # Initialize residual coefficients
              RESI[sysDex] = RHS
              RESCF = computeResidualViscCoeffs(SOLT[:,0], RESI, DX, DZ, udex, OPS)
              
              # Start the time loop
              for tt in range(len(TI)):
                     # Get the DynSGS Coefficients
                     if ResDiff and tt % RTI == 0:
                            RESI[sysDex] = RHS
                            # Compute the local DynSGS coefficients
                            RESCF = computeResidualViscCoeffs(SOLT[:,0], RESI, DX, DZ, udex, OPS)
                     
                     # Compute the SSPRK93 stages
                     sol, RHS = computeTimeIntegrationLN(PHYS, REFS, bN, AN, DT, RHS, SOLT, INIT, RESCF, sysDex, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
                     SOLT[sysDex,0] = sol
                     
                     # Print out diagnostics every OTI steps
                     if tt % OTI == 0:
                            err = np.linalg.norm(RHS)
                            error.append(err)
                            print('Time: ', tt * DT, ' RHS 2-norm: ', err)
                            
                     #if DT * tt >= 1800.0:
                     #       break
              
       elif NonLinSolve:
              sysDex = np.array(range(0, numVar * OPS))
              print('Starting Nonlinear Transient Solver...')
              # Initialize transient storage
              SOLT = np.zeros((numVar * OPS, 2))
              INIT = np.zeros((numVar * OPS,))
              RESI = np.zeros((numVar * OPS,))
              
              # Initialize the solution fields
              INIT[udex] = np.reshape(UZ, (OPS,), order='F')
              INIT[wdex] = np.zeros((OPS,))
              INIT[pdex] = np.reshape(LOGP, (OPS,), order='F')
              INIT[tdex] = np.reshape(LOGT, (OPS,), order='F')
              
              # Get the static vertical gradients and store
              DUDZ = np.reshape(REFS[10], (OPS,), order='F')
              DLPDZ = np.reshape(REFS[11], (OPS,), order='F')
              DLPTDZ = np.reshape(REFS[12], (OPS,), order='F')
              REFG = [DUDZ, DLPDZ, DLPTDZ, ROPS]
              
              if isRestart:
                     rdb = shelve.open('restartDB')
                     SOLT[udex,0] = rdb['uxz']
                     SOLT[wdex,0] = rdb['wxz']
                     SOLT[pdex,0] = rdb['pxz']
                     SOLT[tdex,0] = rdb['txz']
                     RHS = rdb['RHS']
                     NX_in = rdb['NX']
                     NZ_in = rdb['NZ']
                     IT = rdb['ET']
                     rdb.close()
                     
                     if NX_in != NX or NZ_in != NZ:
                            print('ERROR: RESTART DATA IS INVALID')
                            sys.exit(2)
                            
                     if ET <= IT:
                            print('ERROR: END TIME LEQ INITIAL TIME ON RESTART')
                            sys.exit(2)
                            
                     # Initialize the restart time array
                     TI = np.array(np.arange(IT + DT, ET, DT))
              else:
                     # Initialize time array
                     TI = np.array(np.arange(DT, ET, DT))
                     # Initialize the boundary condition
                     SOLT[wbdex,0] = DZT[0,:] * UZ[0,:]
                     # Initialize fields
                     uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, SOLT[:,0], INIT, udex, wdex, pdex, tdex, ubdex, utdex)
                     # Initialize the RHS and forcing for each field
                     RHS = computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, RdT, ubdex, utdex)
                     
              # Initialize residual coefficients
              RESCF = computeResidualViscCoeffs(SOLT[:,0], RHS, DX, DZ, udex, OPS)
              error = [np.linalg.norm(RHS)]
              # Start the time loop
              for tt in range(len(TI)):
                     # Get the DynSGS Coefficients
                     if ResDiff and tt % RTI == 0:
                            # Compute the local DynSGS coefficients
                            RESCF = computeResidualViscCoeffs(SOLT[:,0], RHS, DX, DZ, udex, OPS)
                            
                     # Compute the SSPRK93 stages at this time step
                     sol, RHS = computeTimeIntegrationNL(PHYS, REFS, REFG, DT, RHS, SOLT, INIT, RESCF, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
                     SOLT[sysDex,0] = sol
                     
                     # Print out diagnostics every OTI steps
                     if tt % OTI == 0:
                            err = np.linalg.norm(RHS)
                            error.append(err)
                            print('Time: ', tt * DT, ' Residual 2-norm: ', err)
                            
                     #if DT * tt >= 360:
                     #       break
              
       endt = time.time()
       print('Solve the system: DONE!')
       print('Elapsed time: ', endt - start)
       
       #%% Recover the solution (or check the residual)
       SOL = np.zeros(numVar * OPS)
       SOL[sysDex] = sol;
       SOL[wbdex] = np.multiply(DZT[0,:], np.add(UZ[0,:], SOL[ubdex]))
       
       #% Make a database for restart
       if toRestart:
              rdb = shelve.open('restartDB', flag='n')
              rdb['uxz'] = SOL[udex]
              rdb['wxz'] = SOL[wdex]
              rdb['pxz'] = SOL[pdex]
              rdb['txz'] = SOL[tdex]
              rdb['RHS'] = RHS
              rdb['NX'] = NX
              rdb['NZ'] = NZ
              rdb['ET'] = ET
              rdb.close()
       
       #% Get the fields in physical space
       udex = np.array(range(OPS))
       wdex = np.add(udex, iW * OPS)
       pdex = np.add(udex, iP * OPS)
       tdex = np.add(udex, iT * OPS)
       uxz = np.reshape(SOL[udex], (NZ, NX+1), order='F');
       wxz = np.reshape(SOL[wdex], (NZ, NX+1), order='F');
       pxz = np.reshape(SOL[pdex], (NZ, NX+1), order='F');
       txz = np.reshape(SOL[tdex], (NZ, NX+1), order='F');
       print('Recover solution on native grid: DONE!')
       
       #% Interpolate columns to a finer grid for plotting
       NZI = 200
       uxzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, uxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       wxzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, wxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       pxzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, pxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       txzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, txz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       print('Interpolate columns to finer grid: DONE!')
       
       #% Interpolate rows to a finer grid for plotting
       NXI = 2500
       uxzint = computeHorizontalInterp(DIMS, NXI, uxzint, HF_TRANS)
       wxzint = computeHorizontalInterp(DIMS, NXI, wxzint, HF_TRANS)
       pxzint = computeHorizontalInterp(DIMS, NXI, pxzint, HF_TRANS)
       txzint = computeHorizontalInterp(DIMS, NXI, txzint, HF_TRANS)
       print('Interpolate columns to finer grid: DONE!')
       
       #% Make the new grid XLI, ZTLI
       import HerfunChebNodesWeights as hcnw
       xnew, dummy = hcnw.hefunclb(NX)
       xmax = np.amax(xnew)
       xmin = np.amin(xnew)
       # Make new reference domain grid vectors
       xnew = np.linspace(xmin, xmax, num=NXI, endpoint=True)
       znew = np.linspace(0.0, ZH, num=NZI, endpoint=True)
       
       # Interpolate the terrain profile
       hcf = HF_TRANS.dot(ZTL[0,:])
       dhcf = HF_TRANS.dot(DZT[0,:])
       IHF_TRANS = hcnw.hefuncm(NX, xnew, True)
       hnew = (IHF_TRANS).dot(hcf)
       dhnewdx = (IHF_TRANS).dot(dhcf)
       
       # Scale znew to physical domain and make the new grid
       xnew *= L2 / xmax
       # Compute the new Guellrich domain
       NDIMS = [L1, L2, ZH, NXI-1, NZI]
       NREFS = [xnew, znew]
       XLI, ZTLI, DZTI, sigmaI = computeGuellrichDomain2D(NDIMS, NREFS, hnew, dhnewdx)
       
       #% #Spot check the solution on both grids
       fig = plt.figure()
       ccheck = plt.contourf(XL, ZTL, wxz, 101, cmap=cm.seismic)
       cbar = fig.colorbar(ccheck)
       #plt.xlim(-25000.0, 25000.0)
       #plt.ylim(0.0, 5000.0)
       #
       fig = plt.figure()
       ccheck = plt.contourf(XLI, ZTLI, wxzint, 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       cbar = fig.colorbar(ccheck)
       #plt.xlim(-20000.0, 20000.0)
       #plt.ylim(0.0, 1000.0)
       #plt.yscale('symlog')
       #
       fig = plt.figure()
       plt.plot(XLI[0,:], wxzint[0:2,:].T, XL[0,:], wxz[0:2,:].T)
       plt.xlim(-15000.0, 15000.0)