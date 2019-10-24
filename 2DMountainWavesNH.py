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
import scipy.linalg as dsl
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
#from computeStretchedDomain2D import computeStretchedDomain2D
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeRayleighEquations import computeRayleighEquations
from computeInterpolatedFields import computeInterpolatedFields

# Numerical stuff
from computeEulerEquationsLogPLogT import computeEulerEquationsLogPLogT
from computeEulerEquationsLogPLogT import computeEulerEquationsLogPLogT_NL
from computeTimeIntegration import computePrepareFields
from computeTimeIntegration import computeTimeIntegrationLN
from computeTimeIntegration import computeTimeIntegrationNL
from computeIterativeSolveNL import computeIterativeSolveNL

import faulthandler; faulthandler.enable()

#from matplotlib.animation import ImageMagickWriter

if __name__ == '__main__':
       # Set the solution type
       StaticSolve = True
       LinearSolve = False
       NonLinSolve = False
       ResDiff = False
       
       # Set restarting
       toRestart = True
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
       NX = 131 # FIX: THIS HAS TO BE AN ODD NUMBER!
       NZ = 86
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
       
       #% Transient solve parameters
       DT = 0.1 # Linear transient
       #DT = 0.05 # Nonlinear transient
       HR = 1.0
       ET = HR * 60 * 60 # End time in seconds
       OTI = 200 # Stride for diagnostic output
       ITI = 2000 # Stride for image output
       RTI = 1 # Stride for residual visc update
       
       #% Define the computational and physical grids+
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
       #XL, ZTL, DZT, sigma = computeStretchedDomain2D(DIMS, REFS, HofX, dHdX)
       # Update the REFS collection
       REFS.append(XL)
       REFS.append(ZTL)
       REFS.append(dHdX)
       REFS.append(sigma)
       
       # Compute DX and DZ grid length scales
       DX = np.mean(np.abs(np.diff(REFS[0])))
       DZ = np.mean(np.abs(np.diff(REFS[1])))
       
       #% Compute the BC index vector
       ubdex, utdex, wbdex, sysDex, vbcDex, wbcDex, tbcDex = \
              computeAdjust4CBC(DIMS, numVar, varDex)
       
       #% Read in sensible or potential temperature soundings (corner points)
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
       
       #% Compute the background gradients in physical 2D space
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
       # Compute horizontal derivatives of background fields
       DUDX = np.zeros((NZ,NX+1))
       DLPDX = np.zeros((NZ,NX+1))
       DLTDX = np.zeros((NZ,NX+1))
       for rr in range(NZ):
              # Compute X derivative without constant offsets
              DUDX[rr,:] = DDX_1D.dot(UZ[rr,:] - UZ[rr,0])
              DLPDX[rr,:] = DDX_1D.dot(LOGP[rr,:] - LOGP[rr,0])
              DLTDX[rr,:] = DDX_1D.dot(LOGT[rr,:] - LOGT[rr,0])
              
       # Get the static vertical gradients and store
       DUDX = np.reshape(DUDX, (OPS,), order='F')
       DLPDX = np.reshape(DLPDX, (OPS,), order='F')
       DLPTDX = np.reshape(DLTDX, (OPS,), order='F')
       DUDZ = np.reshape(DUDZ, (OPS,), order='F')
       DLPDZ = np.reshape(DLPDZ, (OPS,), order='F')
       DLPTDZ = np.reshape(DLPTDZ, (OPS,), order='F')
       
       # Make a collection for background field derivatives
       REFG = [DUDX, DLPDX, DLPTDX, DUDZ, DLPDZ, DLPTDZ]
       
       # Update the REFS collection
       REFS.append(UZ)
       REFS.append(PORZ)
       
       # Get some memory back here
       del(PORZ)
       del(DUDZ)
       del(DLPDZ)
       del(DLPTDZ)
       
       #% Get the 2D linear operators...
       DDXM, DDZM = computePartialDerivativesXZ(DIMS, REFS)              
       REFS.append(DDXM)
       REFS.append(DDZM)
       REFS.append(DDXM.dot(DDXM))
       REFS.append(DDZM.dot(DDZM))
       REFS.append(DZT)
       REFS.append(np.reshape(DZT, (OPS,), order='F'))
       del(DDXM)
       del(DDZM)
       
       #% Rayleigh opearator
       ROPS = computeRayleighEquations(DIMS, REFS, mu, depth, width, applyTop, applyLateral, ubdex, utdex)
       REFG.append(ROPS)
       
       #% Compute the global LHS operator
       if StaticSolve or LinearSolve:
              # Compute the equation blocks
              DOPS = computeEulerEquationsLogPLogT(DIMS, PHYS, REFS, REFG)
              print('Compute global sparse linear Euler operator: DONE!')
              # Apply the BC indexing block-wise
              A = DOPS[0]
              B = (DOPS[1].tolil())[:,wbcDex]
              C = DOPS[2]
              D = (DOPS[3])[np.ix_(wbcDex,wbcDex)] 
              E = (DOPS[4])[wbcDex,:]
              F = (DOPS[5].tolil())[np.ix_(wbcDex,tbcDex)]
              G = DOPS[6]
              H = (DOPS[7])[:,wbcDex]
              J = DOPS[8]
              K = (DOPS[9].tolil())[np.ix_(tbcDex,wbcDex)]
              M = (DOPS[3])[np.ix_(tbcDex,tbcDex)]
              R1 = ROPS[0]
              R2 = (ROPS[1].tolil())[np.ix_(wbcDex,wbcDex)]
              R3 = ROPS[2]
              R4 = (ROPS[3].tolil())[np.ix_(tbcDex,tbcDex)]
              
              # Compute the forcing
              WBC = dHdX * UZ[0,:]
              WEQ = sps.bmat([[((DOPS[1].tolil())[:,ubdex])], \
                              [((DOPS[3])[:,ubdex])], \
                              [((DOPS[7])[:,ubdex])], \
                              [((DOPS[9].tolil())[:,ubdex])]])
              bN = -WEQ.dot(WBC); del(WBC)
              
              if StaticSolve and not LinearSolve:
                     # Compute the partitions for Schur Complement solution
                     AS = sps.bmat([[A + R1, B], [None, D + R2]], format='csc')
                     BS = sps.bmat([[C, None], [E, F]], format='csc')
                     CS = sps.bmat([[G, H], [None, K]], format='csc')
                     DS = sps.bmat([[J + R3, None], [None, M + R4]], format='csc')
                     # Get sizes
                     AS_size = AS.shape()
                     DS_size = DS.shape()
                     del(A); del(B); del(C)
                     del(D); del(E); del(F)
                     del(G); del(H); del(J)
                     del(K); del(M)
                     
                     # Compute the partitions for Schur Complement solution
                     fw = bN[wdex]
                     f1 = np.concatenate((bN[udex], fw[wbcDex]))
                     ft = bN[tdex]
                     f2 = np.concatenate((bN[pdex], ft[tbcDex]))
                     del(fw)
                     del(ft)
                     
              if LinearSolve and not StaticSolve:
                     # Compute the global linear operator
                     AN = sps.bmat([[A + R1, B, C, None], \
                              [None, D + R2, E, F], \
                              [G, H, J + R3, None], \
                              [None, K, None, M + R4]], format='csc')
              
                     # Compute the global linear force vector
                     bN = bN[sysDex]
              
              print('Set up global linear operators: DONE!')
       
       #%% Solve the system - Static or Transient Solution
       
       # Initialize transient storage
       SOLT = np.zeros((numVar * OPS, 2))
       INIT = np.zeros((numVar * OPS,))
       
       # Initialize the Background fields
       INIT[udex] = np.reshape(UZ, (OPS,), order='F')
       INIT[wdex] = np.zeros((OPS,))
       INIT[pdex] = np.reshape(LOGP, (OPS,), order='F')
       INIT[tdex] = np.reshape(LOGT, (OPS,), order='F')
       
       # Initialize fields
       fields, uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, SOLT[:,0], INIT, udex, wdex, pdex, tdex, ubdex, utdex)
       
       start = time.time()
       if StaticSolve:
              restart_file = 'restartDB_NL'
              print('Starting Linear to Nonlinear Static Solver...')
              
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
              else:
                     '''
                     #sol = spl.spsolve(AN, bN, permc_spec='MMD_ATA', use_umfpack=False)
                     opts = dict(Equil=True, IterRefine='DOUBLE')
                     factor = spl.splu(AN, permc_spec='MMD_ATA', options=opts)
                     del(AN)
                     sol = factor.solve(bN)
                     del(factor)
                     SOLT[sysDex,0] = sol#[invDex]
                     '''
                     # Factor DS and compute the Schur Complement of DS
                     opts = dict(Equil=True, IterRefine='DOUBLE')
                     factorDS = spl.splu(DS, permc_spec='MMD_ATA', options=opts)
                     # Compute inverse of D
                     IDEN_D = np.identity(DS_size[0])
                     IDS = factorDS.solve(IDEN_D)
                     del(factoDS)
                     print('Factor D matrix... DONE!')
                     # Compute alpha = DS^-1 * CS and f2_hat = DS^-1 * f2
                     alpha = IDS.dot(CS.toarray())
                     f2_hat = IDS.dot(f2)
                     DS_SC = AS.toarray() - (BS.toarray()).dot(alpha)
                     f1_hat = f1 - BS.dot(f2_hat)
                     del(BS)
                     print('Compute Schur Complement of D... DONE!')
                     # Use dense linear algebra at this point
                     sol1 = dsl.solve(DS_SC, f1_hat)
                     del(DS_SC)
                     print('Solve for u and w... DONE!')
                     f2 = f2 - CS.dot(sol1)
                     sol2 = IDS.dot(f2)
                     print('Solve for ln(p) and ln(theta)... DONE!')
                     sol = np.concatenate((sol1, sol2))
                     SOLT[sysDex,0] = sol
                     # Set the boundary condition   
                     SOLT[wbdex,0] = dHdX * UZ[0,:]
                     print('Recover full linear solution vector... DONE!')
                     
                     # Compute and store the LDU decompositions for the inverse
                     IDEN_A = sps.identity(AS_size[0])
                     INV_LDU = [sps.bmat([[IDEN_A, None], [-alpha, IDEN_D]], format='lil'), \
                                sps.bmat([[DS_SC, None], [None, IDS]], format='lil'), \
                                sps.bmat([[IDEN_A, -BS.dot(IDS)], [None, IDEN_D]], format='lil')]
                     
              # Get memory back
              del(f1); del(f2); del(f1_hat); del(f2_hat); del(sol1); del(sol2)
              del(AS); del(BS); del(CS); del(DS)
              del(IDS); del(DS_SC); del(alpha)
              #del(factorDS)
              
              #%% Use the linear solution as the initial guess to the nonlinear solution
              sol = computeIterativeSolveNL(PHYS, REFS, REFG, DX, DZ, SOLT, INIT, udex, wdex, pdex, tdex, ubdex, utdex, sysDex, INV_LDU)
              SOLT[:,1] = sol
              
              # Compare the linear and nonlinear solutions
              DSOL = SOLT[:,1] - SOLT[:,0]
              print('Norm of difference nonlinear to linear solution: ', np.linalg.norm(DSOL))
              
              # Compute and print out the residual
              fields, uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, SOLT[:,1], INIT, udex, wdex, pdex, tdex, ubdex, utdex)
              # Initialize the RHS and forcing for each field
              RHS = computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, ubdex, utdex)
              print('Residual 2-norm: ', np.linalg.norm(RHS))
              #%%
       elif LinearSolve:
              restart_file = 'restartDB_LN'
              print('Starting Linear Transient Solver...')
              
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
                     SOLT[wbdex,0] = dHdX * UZ[0,:]
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
                            # MUST FIX THIS INTERFACE TO EITHER USE THE FULL OPERATOR OR MAKE A MORE EFFICIENT MULTIPLICATION FUNCTION FOR AN
                            sol, rhs = computeTimeIntegrationLN(PHYS, REFS, REFG, bN, AN, DX, DZ, DT, RHS, SOLT, INIT, sysDex, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
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
                            # Check the tendencies
                            #plt.xlim(-30, 30)
                            #plt.ylim(0, 25)
                            for pp in range(4):
                                   plt.subplot(2,2,pp+1)
                                   if pp == 0:
                                          qdex = udex
                                   elif pp == 1:
                                          qdex = wdex
                                   elif pp == 2:
                                          qdex = pdex
                                   else:
                                          qdex = tdex
                                   dqdt = np.reshape(RHS[qdex], (NZ, NX+1), order='F')
                                   ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 101, cmap=cm.seismic)
                                   cbar = plt.colorbar(ccheck, format='%.3e')
                            plt.show()
                            
                     #if DT * tt >= 3600:
                     #       ET = DT * tt
                     #       break
              
              # Set the boundary condition                      
              SOLT[wbdex,0] = np.multiply(dHdX, np.add(UZ[0,:], SOLT[ubdex,0]))
              
       endt = time.time()
       print('Solve the system: DONE!')
       print('Elapsed time: ', endt - start)
       
       #% Make a database for restart
       if toRestart:
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
       dhcf = HF_TRANS.dot(dHdX)
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
              ccheck = plt.contourf(1.0E-3 * XLI, 1.0E-3 * ZTLI, interpLN[1], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30.0, 50.0)
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
              plt.title('Linear - W (m/s)')
              plt.subplot(2,2,3)
              ccheck = plt.contourf(1.0E-3 * XLI, 1.0E-3 * ZTLI, interpNL[1], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30.0, 50.0)
              plt.title('Nonlinear - W (m/s)')
              plt.subplot(1,2,2)
              ccheck = plt.contourf(1.0E-3 * XLI, 1.0E-3 * ZTLI, interpDF[1], 201, cmap=cm.seismic)
              cbar = fig.colorbar(ccheck)
              #plt.contour(1.0E-3 * XLI, 1.0E-3 * ZTLI, interpDF[1], 51, colors='black', linewidths=1.25)
              plt.xlim(-15.0, 25.0)
              plt.ylim(0.0, 30.0)
              plt.title('Difference - W (m/s)')
              plt.tight_layout()
              plt.show()
              
       elif LinearSolve or NonLinSolve:
              fig = plt.figure()
              # 2 X 2 subplot with all fields at the final time
              for pp in range(4):
                     plt.subplot(2,2,pp+1)
                     ccheck = plt.contourf(XLI, ZTLI, interp[pp], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
                     cbar = fig.colorbar(ccheck)
              plt.show()


       #%% #Spot check the solution on both grids
       '''
       fig = plt.figure()
       ccheck = plt.contourf(XL, ZTL, nativeLN[1], 101, cmap=cm.seismic)
       cbar = fig.colorbar(ccheck)
       #plt.xlim(-25000.0, 25000.0)
       #plt.ylim(0.0, 5000.0)
       #
       fig = plt.figure()
       ccheck = plt.contourf(XLI, ZTLI, interpLN[1], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       cbar = fig.colorbar(ccheck)
       #plt.xlim(-20000.0, 20000.0)
       #plt.ylim(0.0, 1000.0)
       #plt.yscale('symlog')
       #
       fig = plt.figure()
       plt.plot(XLI[0,:], (interpLN[1])[0:2,:].T, XL[0,:], (nativeLN[1])[0:2,:].T)
       plt.xlim(-15000.0, 15000.0)
       '''