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
from computeTimeIntegration import computePrepareFields
from computeTimeIntegration import computeTimeIntegrationLN
from computeTimeIntegration import computeTimeIntegrationNL
from computeIterativeSolveNL import computeIterativeSolveNL
from computeInterpolatedFields import computeInterpolatedFields

import faulthandler; faulthandler.enable()

#from matplotlib.animation import ImageMagickWriter

if __name__ == '__main__':
       # Set the solution type
       StaticSolve = True
       LinearSolve = False
       NonLinSolve = False
       ResDiff = False
       
       # Set restarting
       toRestart = False
       isRestart = False
       
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
       NX = 168
       NZ = 96
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
       h0 = 1000.0
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
       DT = 0.005 # Linear transient
       #DT = 0.05 # Nonlinear transient
       HR = 5.0
       ET = HR * 60 * 60 # End time in seconds
       OTI = 200 # Stride for diagnostic output
       ITI = 2000 # Stride for image output
       RTI = 1 # Stride for residual visc update
       
       #%% Define the computational and physical grids+
       REFS = computeGrid(DIMS)
       
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
       
       # Compute DX and DZ grid length scales
       DX = np.amax(np.abs(np.diff(REFS[0])))
       DZ = np.amax(np.abs(np.diff(REFS[1])))
       
       #%% Compute the BC index vector
       ubdex, utdex, wbdex, sysDex, vbcDex = computeAdjust4CBC(DIMS, numVar, varDex)
       
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
       ROPS = computeRayleighEquations(DIMS, REFS, mu, depth, width, applyTop, applyLateral, ubdex, utdex)
       
       #%% Initialize the global solution vector
       SOL = np.zeros((NX * NZ,1))
       
       #%% Rayleigh opearator
       RAYOP = sps.block_diag((ROPS[0], ROPS[1], ROPS[2], ROPS[3]), format='lil')
       
       #%% Compute the global LHS operator
       if StaticSolve or LinearSolve:
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
              AN = A[sysDex,:]
              AN = AN[:,sysDex]
              del(A)
              AN = AN.tocsc()
              bN = -(LDG[:,wbdex]).dot(WBC)
              del(LDG)
              del(WBC)
              print('Set up global solution operators: DONE!')
       
       #%% Solve the system - Static or Transient Solution
       
       # Initialize transient storage
       SOLT = np.zeros((numVar * OPS, 2))
       INIT = np.zeros((numVar * OPS,))
       
       # Initialize the Background fields
       INIT[udex] = np.reshape(UZ, (OPS,), order='F')
       INIT[wdex] = np.zeros((OPS,))
       INIT[pdex] = np.reshape(LOGP, (OPS,), order='F')
       INIT[tdex] = np.reshape(LOGT, (OPS,), order='F')
       
       # Get the static vertical gradients and store
       DUDZ = np.reshape(REFS[10], (OPS,), order='F')
       DLPDZ = np.reshape(REFS[11], (OPS,), order='F')
       DLPTDZ = np.reshape(REFS[12], (OPS,), order='F')
       REFG = [DUDZ, DLPDZ, DLPTDZ, ROPS]
       
       start = time.time()
       if StaticSolve:
              print('Starting Linear to Nonlinear Static Solver...')
              # Make the normal equations
              #AN = (AN.T).dot(AN)
              #bN = (AN.T).dot(bN[sysDex])
              # Solve the system
              #from sksparse.cholmod import cholesky
              #factor = cholesky(AN, ordering_method='colamd'); del(AN)
              #SOLT[sysDex,0] = factor(bN); del(bN)
              bN = bN[sysDex]
              SOLT[sysDex,0] = spl.spsolve(AN, bN, use_umfpack=False)
              # Set the boundary condition                      
              SOLT[wbdex,0] = np.multiply(DZT[0,:], np.add(UZ[0,:], SOLT[ubdex,0]))
              
              # Get the static vertical gradients and store
              DUDZ = np.reshape(REFS[10], (OPS,), order='F')
              DLPDZ = np.reshape(REFS[11], (OPS,), order='F')
              DLPTDZ = np.reshape(REFS[12], (OPS,), order='F')
              REFG = [DUDZ, DLPDZ, DLPTDZ, ROPS]
              
              #%% Use the linear solution as the initial guess to the nonlinear solution
              sol = computeIterativeSolveNL(PHYS, REFS, REFG, DX, DZ, SOLT[:,0], INIT, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
              SOLT[:,1] = sol
              
              # Compare the linear and nonlinear solutions
              DSOL = SOLT[:,1] - SOLT[:,0]
              print('Norm of difference nonlinear to linear solution: ', np.linalg.norm(DSOL))
              #%%
       elif LinearSolve:
              restart_file = 'restartDB_LN'
              print('Starting Linear Transient Solver...')
              bN = bN[sysDex]
              
              if isRestart:
                     rdb = shelve.open(restart_file, flag='r')
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
                     # Initialize the RHS
                     RHS = np.zeros((numVar * OPS,))
                     RHS[sysDex] = bN
              
       elif NonLinSolve:
              restart_file = 'restartDB_NL'
              sysDex = np.array(range(0, numVar * OPS))
              print('Starting Nonlinear Transient Solver...')
              
              if isRestart:
                     rdb = shelve.open(restart_file)
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
                     fields, uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, SOLT[:,0], INIT, udex, wdex, pdex, tdex, ubdex, utdex)
                     # Initialize the RHS and forcing for each field
                     RHS = computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, ubdex, utdex)
                                          
       #%% Start the time loop
       if LinearSolve or NonLinSolve:
              error = [np.linalg.norm(RHS)]
              #metadata = dict(title='Nonlinear Solve - Ln-Theta, 100 m', artist='Spectral Methond', comment='DynSGS')
              #writer = ImageMagickWriter(fps=20, metadata=metadata)
              fig = plt.figure()
              plt.show()
              #with writer.saving(fig, "nonlinear_dynsgs.gif", 100):
              for tt in range(len(TI)):
                     # Compute the SSPRK93 stages at this time step
                     if LinearSolve:
                            sol, rhs = computeTimeIntegrationLN(PHYS, REFS, bN, AN, DX, DZ, DT, RHS, SOLT, INIT, sysDex, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
                     elif NonLinSolve:
                            sol, rhs = computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, RHS, SOLT, INIT, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
                     
                     SOLT[sysDex,0] = sol
                     RHS[sysDex] = rhs
                     
                     # Print out diagnostics every OTI steps
                     if tt % OTI == 0:
                            err = np.linalg.norm(RHS)
                            error.append(err)
                            print('Time: ', tt * DT, ' Residual 2-norm: ', err)
                     
                     if tt % ITI == 0:
                            # Make animation for check
                            txz = np.reshape(SOLT[tdex,0], (NZ, NX+1), order='F')
                            ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, txz, 101, cmap=cm.jet)
                            plt.xlim(-30, 30)
                            plt.ylim(0, 25)
                            plt.show()
                            
                     #if DT * tt >= 3600:
                     #       ET = DT * tt
                     #       break
              
              # Set the boundary condition                      
              SOLT[wbdex,0] = np.multiply(DZT[0,:], np.add(UZ[0,:], SOLT[ubdex,0]))
              
       endt = time.time()
       print('Solve the system: DONE!')
       print('Elapsed time: ', endt - start)
       
       #% Make a database for restart
       if toRestart and not StaticSolve:
              rdb = shelve.open(restart_file, flag='n')
              rdb['uxz'] = SOLT[udex,0]
              rdb['wxz'] = SOLT[wdex,0]
              rdb['pxz'] = SOLT[pdex,0]
              rdb['txz'] = SOLT[tdex,0]
              rdb['RHS'] = RHS
              rdb['NX'] = NX
              rdb['NZ'] = NZ
              rdb['ET'] = ET
              rdb.close()
       
       #%% Recover the solution (or check the residual)
       NXI = 2500
       NZI = 200
       if StaticSolve:
              nativeLN, interpLN = computeInterpolatedFields(DIMS, ZTL, SOLT[:,0], NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
              nativeNL, interpNL = computeInterpolatedFields(DIMS, ZTL, SOLT[:,1], NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
              nativeDF, interpDF = computeInterpolatedFields(DIMS, ZTL, DSOL, NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
       else:
              native, interp = computeInterpolatedFields(DIMS, ZTL, SOLT[:,0], NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
              uxz = native[0]; wxz = native[1]; pxz = native[2]; txz = native[3]
              uxzint = interp[0]; wxzint = interp[1]; pxzint = interp[2]; txzint = interp[3]
       
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
       
       #%% Make some plots for static or transient solutions
       
       if StaticSolve:
              fig = plt.figure(figsize=(12.0, 6.0))
              # 1 X 3 subplot of W for linear, nonlinear, and difference
              plt.subplot(2,2,1)
              ccheck = plt.contourf(XLI, ZTLI, interpLN[1], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30000.0, 50000.0)
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
              plt.title('Linear - W (m/s)')
              plt.subplot(2,2,3)
              ccheck = plt.contourf(1.0E-3 * XLI, ZTLI, interpNL[1], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30.0, 50.0)
              plt.title('Nonlinear - W (m/s)')
              plt.subplot(1,2,2)
              ccheck = plt.contourf(1.0E-3 * XLI, ZTLI, interpDF[1], 201, cmap=cm.flag)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30.0, 50.0)
              plt.title('Difference - W (m/s)')
              plt.tight_layout()
              
       elif LinearSolve or NonLinSolve:
              fig = plt.figure()
              # 2 X 2 subplot with all fields at the final time
              for pp in range(4):
                     plt.subplot(2,2,pp+1)
                     ccheck = plt.contourf(XLI, ZTLI, interp[pp], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
                     cbar = fig.colorbar(ccheck)
       
       '''
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
       '''