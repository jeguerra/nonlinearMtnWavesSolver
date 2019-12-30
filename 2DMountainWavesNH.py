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
from computeTopographyOnGrid import computeTopographyOnGrid
from computeGuellrichDomain2D import computeGuellrichDomain2D
#from computeStretchedDomain2D import computeStretchedDomain2D
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeRayleighEquations import computeRayleighEquations
from computeInterpolatedFields import computeInterpolatedFields

# Numerical stuff
import computeSchurSolve as sch
import computeDerivativeMatrix as derv
import computeEulerEquationsLogPLogT as eqs
import computeSchurSolve as sch
from computeTimeIntegration import computeTimeIntegrationLN
from computeTimeIntegration import computeTimeIntegrationNL
#from computeIterativeSolveNL import computeIterativeSolveNL

import faulthandler; faulthandler.enable()

def displayResiduals(message, RHS, thisTime, udex, wded, pdex, tdex):
       err = np.linalg.norm(RHS)
       err1 = np.linalg.norm(RHS[udex])
       err2 = np.linalg.norm(RHS[wdex])
       err3 = np.linalg.norm(RHS[pdex])
       err4 = np.linalg.norm(RHS[tdex])
       if message != '':
              print(message)
       print('Time: %d, Residuals: %10.4E, %10.4E, %10.4E, %10.4E, %10.4E' \
             % (thisTime, err1, err2, err3, err4, err))
       
       return err

def getFromRestart(name, ET, NX, NZ, StaticSolve):
       rdb = shelve.open(restart_file, flag='r')
       
       NX_in = rdb['NX']
       NZ_in = rdb['NZ']
       if NX_in != NX or NZ_in != NZ:
              print('ERROR: RESTART DATA IS INVALID')
              print(NX, NX_in)
              print(NZ, NZ_in)
              sys.exit(2)
       
       SOLT = rdb['SOLT']
       LMS = rdb['LMS']
       RHS = rdb['RHS']
       IT = rdb['ET']
       if ET <= IT and not StaticSolve:
              print('ERROR: END TIME LEQ INITIAL TIME ON RESTART')
              sys.exit(2)
              
       # Initialize the restart time array
       TI = np.array(np.arange(IT + DT, ET, DT))
       rdb.close()
       
       return SOLT, LMS, RHS, NX_in, NZ_in, TI

# Store a matrix to disk in column wise chucks
def storeColumnChunks(MM, Mname, dbName):
       # Set up storage and store full array
       mdb = shelve.open(dbName, flag='n')
       # Get the number of cpus
       import multiprocessing as mtp
       NCPU = mtp.cpu_count()
       # Partition CS into NCPU column wise chuncks
       NC = MM.shape[1] # Number of columns in MM
       RC = NC % NCPU # Remainder of columns when dividing by NCPU
       SC = int((NC - RC) / NCPU) # Number of columns in each chunk
       
       # Loop over NCPU column chunks and store
       cranges = []
       for cc in range(NCPU):
              cbegin  = cc * SC
              if cc < NCPU - 1:
                     crange = range(cbegin,cbegin + SC)
              elif cc == NCPU - 1:
                     crange = range(cbegin,cbegin + SC + RC)
              
              cranges.append(crange)
              mdb[Mname + str(cc)] = MM[:,crange]
              
       mdb.close()
              
       return NCPU, cranges

def computeSchurBlock(dbName, blockName):
       # Open the blocks database
       bdb = shelve.open(dbName, flag='r')
       
       if blockName == 'AS':
              SB = sps.bmat([[bdb['A'], bdb['B']], \
                             [bdb['E'], bdb['F']]], format='csc')
       elif blockName == 'BS':
              SB = sps.bmat([[bdb['C'], bdb['D'], bdb['LDA']], \
                             [bdb['G'], bdb['H'], bdb['LHA']]], format='csc')
       elif blockName == 'CS':
              SB = sps.bmat([[bdb['I'], bdb['J']], \
                             [bdb['N'], bdb['O']], \
                             [bdb['LNA'], bdb['LOA']]], format='csc')
       elif blockName == 'DS':
              SB = sps.bmat([[bdb['K'], bdb['M'], bdb['LMA']], \
                             [bdb['P'], bdb['Q'], bdb['LQAC']], \
                             [bdb['LPA'], bdb['LQAR'], bdb['LDIA']]], format='csc')
       else:
              print('INVALID SCHUR BLOCK NAME!')
              
       bdb.close()
       return SB.toarray()
       
if __name__ == '__main__':
       # Set the solution type (MUTUALLY EXCLUSIVE)
       StaticSolve = True
       LinearSolve = False
       NonLinSolve = False
       
       # Set the grid type
       HermCheb = True
       UniformDelta = False
       
       # Set residual diffusion switch
       ResDiff = False
       
       # Set direct solution method (MUTUALLY EXCLUSIVE)
       SolveFull = False
       SolveSchur = True
       
       # Set Newton solve initial and restarting parameters
       toRestart = True # Saves resulting state to restart database
       isRestart = False # Initializes from a restart database
       #localDir = '/media/jeguerra/DATA/scratch/'
       localDir = '/scratch/'
       restart_file = localDir + 'restartDB'
       schurName = localDir + 'SchurOps'
       
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
       NX = 147 # FIX: THIS HAS TO BE AN ODD NUMBER!
       NZ = 92
       OPS = (NX + 1) * NZ
       numVar = 4
       NQ = OPS * numVar
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
       
       # Set the background temperature profile
       T_in = [300.0, 228.5, 228.5, 248.5]
       Z_in = [0.0, 1.1E4, 2.0E4, ZH]
       
       # Set the terrain options
       h0 = 100.0
       aC = 5000.0
       lC = 4000.0
       HOPT = [h0, aC, lC]
       
       # Set the Rayleigh options
       depth = 12000.0
       width = 24000.0
       applyTop = True
       applyLateral = True
       mu = np.array([1.0E-2, 1.0E-2, 1.0E-2, 1.0E-2])
       mu *= 1.0
       
       #% Transient solve parameters
       DT = 0.05
       HR = 5.0
       rampTime = 900  # 10 minutes to ramp up U_bar
       intMethodOrder = 3 # 3rd or 4th order time integrator
       ET = HR * 60 * 60 # End time in seconds
       OTI = 200 # Stride for diagnostic output
       ITI = 1000 # Stride for image output
       RTI = 1 # Stride for residual visc update
       
       #% Define the computational and physical grids+
       REFS = computeGrid(DIMS, HermCheb, UniformDelta)
       
       #% Compute the raw derivative matrix operators in alpha-xi computational space
       DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
       
       DDX_SP = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[0])
       DDZ_SP = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[1])
       
       # Update the REFS collection
       REFS.append(DDX_1D)
       REFS.append(DDZ_1D)
       
       #% Read in topography profile or compute from analytical function
       KAISER = 1 # Kaiser window profile
       SCHAR = 2 # Schar mountain profile nominal (Schar, 2001)
       EXPCOS = 3 # Even exponential and squared cosines product
       EXPPOL = 4 # Even exponential and even polynomial product
       INFILE = 5 # Data from a file (equally spaced points)
       HofX, dHdX = computeTopographyOnGrid(REFS, SCHAR, HOPT, width)
       
       # Make the 2D physical domains from reference grids and topography
       zRay = ZH - depth
       XL, ZTL, DZT, sigma, ZRL = computeGuellrichDomain2D(DIMS, REFS, zRay, HofX, dHdX)
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
       ubdex, utdex, wbdex, pbdex, tbdex, \
              ubcDex, wbcDex, pbcDex, tbcDex, \
              zeroDex_stat, zeroDex_tran, sysDex = \
              computeAdjust4CBC(DIMS, numVar, varDex)
       
       #% Read in sensible or potential temperature soundings (corner points)
       SENSIBLE = 1
       POTENTIAL = 2
       # Map the sounding to the computational vertical 2D grid [0 H]
       TZ, DTDZ = computeTemperatureProfileOnGrid(Z_in, T_in, REFS, True)
       
       # Compute background fields on the reference column
       dlnPdz, LPZ, PZ, dlnPTdz, LPT, PT, RHO = \
              computeThermoMassFields(PHYS, DIMS, REFS, TZ[:,0], DTDZ[:,0], SENSIBLE)
       
       # Read in or compute background horizontal wind profile
       MEANJET = 1 # Analytical smooth jet profile
       JETOPS = [10.0, 16.822, 1.386]

       U, dUdz = computeShearProfileOnGrid(REFS, JETOPS, P0, PZ, dlnPdz)
       
       #% Compute the background gradients in physical 2D space
       dUdz = np.expand_dims(dUdz, axis=1)
       DUDZ = np.tile(dUdz, NX+1)
       DUDZ = computeColumnInterp(DIMS, REFS[1], dUdz, 0, ZTL, DUDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       # Compute thermodynamic gradients (no interpolation!)
       PORZ = Rd * TZ
       DLPDZ = -gc / Rd * np.reciprocal(TZ)
       DLTDZ = np.reciprocal(TZ) * DTDZ
       DLPTDZ = DLTDZ - Kp * DLPDZ
       
       # Compute the background (initial) fields
       U = np.expand_dims(U, axis=1)
       UZ = np.tile(U, NX+1)
       UZ = computeColumnInterp(DIMS, REFS[1], U, 0, ZTL, UZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       LPZ = np.expand_dims(LPZ, axis=1)
       LOGP = np.tile(LPZ, NX+1)
       LOGP = computeColumnInterp(DIMS, REFS[1], LPZ, 0, ZTL, LOGP, CH_TRANS, '1DtoTerrainFollowingCheb')
       LPT = np.expand_dims(LPT, axis=1)
       LOGT = np.tile(LPT, NX+1)
       LOGT = computeColumnInterp(DIMS, REFS[1], LPT, 0, ZTL, LOGT, CH_TRANS, '1DtoTerrainFollowingCheb')
         
       # Get the static vertical gradients and store
       DUDZ = np.reshape(DUDZ, (OPS,1), order='F')
       DLTDZ = np.reshape(DLTDZ, (OPS,1), order='F')
       DLPDZ = np.reshape(DLPDZ, (OPS,1), order='F')
       DLPTDZ = np.reshape(DLPTDZ, (OPS,1), order='F')
       DQDZ = np.hstack((DUDZ, np.zeros((OPS,1)), DLPDZ, DLPTDZ))
       
       # Make a collection for background field derivatives
       REFG = [DUDZ, DLTDZ, DLPDZ, DLPTDZ, DQDZ]
       
       # Update the REFS collection
       REFS.append(np.reshape(UZ, (OPS,1), order='F'))
       REFS.append(np.reshape(PORZ, (OPS,1), order='F'))
       
       # Get some memory back here
       del(PORZ)
       del(DUDZ)
       del(DLTDZ)
       del(DLPDZ)
       del(DLPTDZ)
       
       #%% Rayleigh opearator and GML weight
       ROPS, GMLX, GMLZ = computeRayleighEquations(DIMS, REFS, mu, ZRL, width, applyTop, applyLateral, ubdex, utdex)
       REFG.append(ROPS)
       GMLXOP = sps.diags(np.reshape(GMLX, (OPS,), order='F'), offsets=0, format='csr')
       GMLZOP = sps.diags(np.reshape(GMLZ, (OPS,), order='F'), offsets=0, format='csr')
       del(GMLX); del(GMLZ)
       
       #%% Get the 2D linear operators in Hermite-Chebyshev space
       DDXM, DDZM = computePartialDerivativesXZ(DIMS, REFS, DDX_1D, DDZ_1D)
       DZDX = sps.diags(np.reshape(DZT, (OPS,), order='F'), offsets=0, format='csr')
       
       #%% Get the 2D linear operators in Compact Finite Diff (for Laplacian)
       DDXM_SP, DDZM_SP = computePartialDerivativesXZ(DIMS, REFS, DDX_SP, DDZ_SP)
       
       # Update the data storage
       REFS.append(GMLXOP.dot(DDXM))
       REFS.append(GMLZOP.dot(DDZM))
       # Store sparse derivatives
       REFS.append(GMLXOP.dot(DDXM_SP))
       REFS.append(GMLZOP.dot(DDZM_SP))
       REFS.append(DZT)
       REFS.append(DZDX.diagonal())
       
       del(DDXM)
       del(DDZM)
       #del(DDXM_SP)
       #del(DDZM_SP)
       del(DZDX)
       
       #%% SOLUTION INITIALIZATION
       physDOF = numVar * OPS
       totalDOF = physDOF + NX
       
       # Initialize hydrostatic background
       INIT = np.zeros((physDOF,))
       RHS = np.zeros((physDOF,))
       
       # Initialize the Background fields
       INIT[udex] = np.reshape(UZ, (OPS,), order='F')
       INIT[wdex] = np.zeros((OPS,))
       INIT[pdex] = np.reshape(LOGP, (OPS,), order='F')
       INIT[tdex] = np.reshape(LOGT, (OPS,), order='F')
       
       if isRestart:
              print('Restarting from previous solution...')
              SOLT, LMS, RHS, NX_in, NZ_in, TI = getFromRestart(restart_file, ET, NX, NZ, StaticSolve)
              
              # Updates nolinear boundary condition to next Newton iteration
              dWBC = SOLT[wbdex,0] - dHdX * (INIT[ubdex] + SOLT[ubdex,0])
       else:
              # Initialize solution storage
              SOLT = np.zeros((physDOF, 2))
              
              # Initialize Lagrange Multiplier storage
              LMS = np.zeros(NX)
              
              # Initial change in vertical velocity at boundary
              dWBC = -dHdX * INIT[ubdex]
       
              # Initialize time array
              TI = np.array(np.arange(DT, ET, DT))
            
       # Prepare the current fields (TO EVALUATE CURRENT JACOBIAN)
       currentState = np.array(SOLT[:,0])
       fields, U, RdT = eqs.computePrepareFields(PHYS, REFS, currentState, INIT, udex, wdex, pdex, tdex)
              
       #% Compute the global LHS operator and RHS
       if (StaticSolve or LinearSolve):
              
              # SET THE BOOLEAN ARGUMENT TO isRestart WHEN USING DISCONTINUOUS BOUNDARY DATA
              DOPS_NL = eqs.computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, \
                            np.array(fields), U, RdT, LMS, ubdex, utdex)
              #DOPS = eqs.computeEulerEquationsLogPLogT(DIMS, PHYS, REFS, REFG)

              print('Compute Jacobian operator blocks: DONE!')
              
              # Convert blocks to 'lil' format for efficient indexing
              DOPS = []
              for dd in range(len(DOPS_NL)):
                     if (DOPS_NL[dd]) is not None: 
                            DOPS.append(DOPS_NL[dd].tolil())
                     else:
                            DOPS.append(DOPS_NL[dd])
              del(DOPS_NL)
              
              #'''
              # USE THIS TO SET THE FORCING WITH DISCONTINUOUS BOUNDARY DATA
              RHS = eqs.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, \
                            np.array(fields), U, RdT)
              RHS += eqs.computeRayleighTendency(REFG, np.array(fields)) 
              RHS[zeroDex_stat] *= 0.0
              err = displayResiduals('Current function evaluation residual: ', RHS, 0.0, udex, wdex, pdex, tdex)
              del(U); del(fields)
              
              # Compute forcing vector adding boundary forcing to the end
              bN = np.concatenate((RHS, np.zeros(NX)))
              
              # Compute Lagrange multiplier row augmentation matrices (exclude left corner node)
              R1 = sps.diags(-LMS * dHdX[1:], offsets=0, format='lil')
              R2 = sps.diags(LMS, offsets=0, format='lil')
              C1 = sps.diags(dWBC[1:], offsets=0, format='lil')
              
              rowShape = (NX,OPS)
              LN = sps.lil_matrix(rowShape)
              LN[:,ubdex[1:]] = R1
              LO = sps.lil_matrix(rowShape)
              LO[:,ubdex[1:]] = R2
              LP = sps.lil_matrix(rowShape)
              LQ = sps.lil_matrix(rowShape)
              
              # Apply BC adjustments and indexing block-wise (Lagrange blocks)
              LNA = LN[:,ubcDex]
              LOA = LO[:,wbcDex]
              LPA = LP[:,pbcDex]
              LQAR = LQ[:,tbcDex]
              
              # Compute Lagrange multiplier column augmentation matrices (exclude left corner node)
              colShape = (OPS,NX)
              LD = sps.lil_matrix(colShape)
              LD[ubdex[1:],:] = C1
              LH = sps.lil_matrix(colShape)
              LH[ubdex[1:],:] = C1
              LM = sps.lil_matrix(colShape)
              LM[ubdex[1:],:] = C1
              LQ = sps.lil_matrix(colShape)
              LH[ubdex[1:],:] = C1
              
              # Apply BC adjustments and indexing block-wise (Lagrange blocks)
              LDA = LD[ubcDex,:]
              LHA = LH[wbcDex,:]
              LMA = LM[pbcDex,:]
              LQAC = LQ[tbcDex,:]
              
              # Diagonal corner block
              LDIA = C1
              
              # Apply BC adjustments and indexing block-wise (LHS operator)
              A = DOPS[0][np.ix_(ubcDex,ubcDex)]              
              B = DOPS[1][np.ix_(ubcDex,wbcDex)]
              C = DOPS[2][np.ix_(ubcDex,pbcDex)]
              D = DOPS[3][np.ix_(ubcDex,tbcDex)]
              
              E = DOPS[4][np.ix_(wbcDex,ubcDex)]
              F = DOPS[5][np.ix_(wbcDex,wbcDex)] 
              G = DOPS[6][np.ix_(wbcDex,pbcDex)]
              H = DOPS[7][np.ix_(wbcDex,tbcDex)]
              
              I = DOPS[8][np.ix_(pbcDex,ubcDex)]
              J = DOPS[9][np.ix_(pbcDex,wbcDex)]
              K = DOPS[10][np.ix_(pbcDex,pbcDex)]
              M = DOPS[11] # Block of None
              
              N = DOPS[12][np.ix_(tbcDex,ubcDex)]
              O = DOPS[13][np.ix_(tbcDex,wbcDex)]
              P = DOPS[14] # Block of None
              Q = DOPS[15][np.ix_(tbcDex,tbcDex)]
              
              # The Rayleigh operators are block diagonal
              R1 = (ROPS[0].tolil())[np.ix_(ubcDex,ubcDex)]
              R2 = (ROPS[1].tolil())[np.ix_(wbcDex,wbcDex)]
              R3 = (ROPS[2].tolil())[np.ix_(pbcDex,pbcDex)]
              R4 = (ROPS[3].tolil())[np.ix_(tbcDex,tbcDex)]
               
              del(DOPS)
              
              # Set up Schur blocks or full operator...
              if (StaticSolve and SolveSchur) and not LinearSolve:
                     # Add Rayleigh damping terms
                     A += R1
                     F += R2
                     K += R3
                     Q += R4
                     
                     # Store the operators...
                     opdb = shelve.open(schurName, flag='n')
                     opdb['A'] = A; opdb['B'] = B; opdb['C'] = C; opdb['D'] = D
                     opdb['E'] = E; opdb['F'] = F; opdb['G'] = G; opdb['H'] = H
                     opdb['I'] = I; opdb['J'] = J; opdb['K'] = K; opdb['M'] = M
                     opdb['N'] = N; opdb['O'] = O; opdb['P'] = P; opdb['Q'] = Q
                     opdb['N'] = N; opdb['O'] = O; opdb['P'] = P; opdb['Q'] = Q
                     opdb['LDA'] = LDA; opdb['LHA'] = LHA; opdb['LMA'] = LMA; opdb['LQAC'] = LQAC
                     opdb['LNA'] = LNA; opdb['LOA'] = LOA; opdb['LPA'] = LPA; opdb['LQAR'] = LQAR
                     opdb['LDIA'] = LDIA
                     opdb.close()
                      
                     # Compute the partitions for Schur Complement solution
                     fu = bN[udex]
                     fw = bN[wdex]
                     f1 = np.concatenate((fu[ubcDex], fw[wbcDex]))
                     fp = bN[pdex]
                     ft = bN[tdex]
                     f2 = np.concatenate((fp[pbcDex], ft[tbcDex], dWBC[1:]))
                     del(bN)
                     
              if LinearSolve or (StaticSolve and SolveFull):
                     # Add Rayleigh damping terms
                     A += R1
                     F += R2
                     K += R3
                     Q += R4
                     
                     # Compute the global linear operator
                     AN = sps.bmat([[A, B, C, D], \
                              [E, F, G, H], \
                              [I, J, K, M], \
                              [N, O, P, Q]], format='csr')
              
                     # Compute the global linear force vector
                     bN = bN[sysDex]
              
              # Get memory back
              del(A); del(B); del(C); del(D)
              del(E); del(F); del(G); del(H)
              del(I); del(J); del(K); del(M)
              del(N); del(O); del(P); del(Q)
              print('Set up global linear operators: DONE!')
       
       #%% Solve the system - Static or Transient Solution
       start = time.time()
       if StaticSolve:
              print('Starting Linear to Nonlinear Static Solver...')
              
              if SolveFull and not SolveSchur:
                     print('Solving linear system by full operator SuperLU...')
                     # Direct solution over the entire operator (better for testing BC's)
                     #sol = spl.spsolve(AN, bN, permc_spec='MMD_ATA', use_umfpack=False)
                     opts = dict(Equil=True, IterRefine='DOUBLE')
                     factor = spl.splu(AN, permc_spec='MMD_ATA', options=opts)
                     del(AN)
                     sol = factor.solve(bN)
                     del(bN)
                     del(factor)
              if SolveSchur and not SolveFull:
                     print('Solving linear system by Schur Complement...')
                     # Factor DS and compute the Schur Complement of DS
                     DS = computeSchurBlock(schurName,'DS')
                     factorDS = dsl.lu_factor(DS)
                     del(DS)
                     print('Factor D... DONE!')
                     
                     # Compute f2_hat = DS^-1 * f2 and f1_hat
                     BS = computeSchurBlock(schurName,'BS')
                     f2_hat = dsl.lu_solve(factorDS, f2)
                     f1_hat = f1 - BS.dot(f2_hat)
                     del(BS); del(f2_hat)
                     print('Compute modified force vectors... DONE!')
                     
                     # Get CS block and store in column chunks
                     CS = computeSchurBlock(schurName, 'CS')
                     fileCS = localDir + 'CS'
                     NCPU, cranges = storeColumnChunks(CS, 'CS', fileCS)
                     del(CS)
                     
                     # Loop over the chunks from disk
                     AS = computeSchurBlock(schurName, 'AS')
                     BS = computeSchurBlock(schurName, 'BS')
                     mdb = shelve.open(fileCS, flag='r')
                     for cc in range(NCPU):
                            crange = cranges[cc] 
                            CS_chunk = mdb['CS' + str(cc)]
                            
                            DS_chunk = dsl.lu_solve(factorDS, CS_chunk) # LONG EXECUTION
                            del(CS_chunk)
                            AS[:,crange] -= BS.dot(DS_chunk) # LONG EXECUTION
                            del(DS_chunk)
                            
                     mdb.close()
                     del(BS)
                     print('Solve DS^-1 * CS... DONE!')
                     print('Compute Schur Complement of D... DONE!')
                     
                     # Apply Schur C. solver on block partitioned DS_SC
                     #sol1 = sch.computeSchurSolve(AS, f1_hat)
                     factorDS_SC = dsl.lu_factor(AS)
                     del(AS)
                     print('Factor D and Schur Complement of D... DONE!')
                     
                     sol1 = dsl.lu_solve(factorDS_SC, f1_hat)
                     del(factorDS_SC)
                     print('Solve for u and w... DONE!')
                     
                     CS = computeSchurBlock(schurName, 'CS')
                     f2_hat = f2 - CS.dot(sol1)
                     del(CS)
                     sol2 = dsl.lu_solve(factorDS, f2_hat)
                     del(factorDS)
                     print('Solve for ln(p) and ln(theta)... DONE!')
                     dsol = np.concatenate((sol1, sol2))
                     
                     # Get memory back
                     del(f1); del(f2)
                     del(f1_hat); del(f2_hat)
                     del(sol1); del(sol2)
                     
              #%% Update the interior and boundary solution
              SL = len(dsol)
              dsolQ = dsol[0:(SL - NX)]
              SOLT[sysDex,0] += dsolQ
              # Store solution change to instance 1
              SOLT[sysDex,1] = dsolQ
              # Store the Lagrange Multipliers
              LMS += dsol[(SL - NX):SL]
              
              print('Recover full linear solution vector... DONE!')
              
              #%% Check the output residual
              fields, U, RdT = eqs.computePrepareFields(PHYS, REFS, np.array(SOLT[:,0]), INIT, udex, wdex, pdex, tdex)
              
              # Set the output residual and check
              message = 'Residual 2-norm BEFORE Newton step:'
              err = displayResiduals(message, RHS, 0.0, udex, wdex, pdex, tdex)
              RHS = eqs.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, np.array(fields), U, RdT)
              RHS += eqs.computeRayleighTendency(REFG, np.array(fields))
              message = 'Residual 2-norm AFTER Newton step:'
              err = displayResiduals(message, RHS, 0.0, udex, wdex, pdex, tdex)
              
              # Check the change in the solution
              DSOL = np.array(SOLT[:,1])
              print('Norm of change in solution: ', np.linalg.norm(DSOL))
       #%% Transient solutions       
       elif LinearSolve:
              RHS[sysDex] = bN
              print('Starting Linear Transient Solver...')
       elif NonLinSolve:
              #sysDex = np.array(range(0, numVar * OPS))
              print('Starting Nonlinear Transient Solver...')
                                          
       #%% Start the time loop
       if LinearSolve or NonLinSolve:
              error = [np.linalg.norm(RHS)]
              for tt in range(len(TI)):
                     # Print out diagnostics every OTI steps
                     if tt % OTI == 0:
                            message = ''
                            thisTime = DT * tt
                            err = displayResiduals(message, RHS, thisTime, udex, wdex, pdex, tdex)
                            error.append(err)
                     
                     if tt % ITI == 0:
                            fig = plt.figure(figsize=(10.0, 6.0))
                            # Check the tendencies
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
                     
                     # Ramp up the background wind to decrease transients
                     if not isRestart:
                            if thisTime <= rampTime:
                                   uRamp = 0.5 * (1.0 - mt.cos(mt.pi / rampTime * thisTime))
                                   UT = uRamp * INIT[udex]
                            else:
                                   UT = INIT[udex]
                     else:
                            UT = INIT[udex]
                            
                     # Compute the SSPRK93 stages at this time step
                     if LinearSolve:
                            # MUST FIX THIS INTERFACE TO EITHER USE THE FULL OPERATOR OR MAKE A MORE EFFICIENT MULTIPLICATION FUNCTION FOR AN
                            sol, rhs = computeTimeIntegrationLN(PHYS, REFS, REFG, bN, AN, DX, DZ, DT, RHS, SOLT, INIT, sysDex, udex, wdex, pdex, tdex, ubdex, utdex, ResDiff)
                     elif NonLinSolve:
                            sol, rhs = computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, RHS, SOLT, INIT, zeroDex_tran, udex, wdex, pdex, tdex, ubdex, utdex, wbdex, ResDiff, intMethodOrder)
                     
                     SOLT[:,0] = sol
                     SOLT[wbdex,0] = dHdX * (UT[ubdex] + SOLT[ubdex,0])
                     RHS[:] = rhs
              
              # Copy state instance 0 to 1
              SOLT[:,1] = np.array(SOLT[:,0])
              DSOL = SOLT[:,1] - SOLT[:,0]
       #%%       
       endt = time.time()
       print('Solve the system: DONE!')
       print('Elapsed time: ', endt - start)
       
       #% Make a database for restart
       if toRestart:
              rdb = shelve.open(restart_file, flag='n')
              rdb['DSOL'] = DSOL
              rdb['SOLT'] = SOLT
              rdb['LMS'] = LMS
              rdb['RHS'] = RHS
              rdb['NX'] = NX
              rdb['NZ'] = NZ
              rdb['ET'] = ET
              rdb.close()
       
       #%% Recover the solution (or check the residual)
       NXI = 2500
       NZI = 200
       nativeLN, interpLN = computeInterpolatedFields(DIMS, ZTL, np.array(SOLT[:,0]), NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
       nativeNL, interpNL = computeInterpolatedFields(DIMS, ZTL, np.array(SOLT[:,1]), NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
       nativeDF, interpDF = computeInterpolatedFields(DIMS, ZTL, DSOL, NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS)
       
       uxz = nativeLN[0]; wxz = nativeLN[1]; pxz = nativeLN[2]; txz = nativeLN[3]
       uxzint = interpLN[0]; wxzint = interpLN[1]; pxzint = interpLN[2]; txzint = interpLN[3]
       
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
       XLI, ZTLI, DZTI, sigmaI, ZRLI = computeGuellrichDomain2D(NDIMS, NREFS, zRay, hnew, dhnewdx)
       
       #%% Make some plots for static or transient solutions
       
       if StaticSolve:
              fig = plt.figure(figsize=(12.0, 6.0))
              # 1 X 3 subplot of W for linear, nonlinear, and difference
              
              plt.subplot(2,2,1)
              ccheck = plt.contourf(1.0E-3 * XLI, 1.0E-3 * ZTLI, interpDF[0], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30.0, 50.0)
              plt.ylim(0.0, 30.0)
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
              plt.title('Change U - (m/s/s)')
              
              plt.subplot(2,2,3)
              ccheck = plt.contourf(1.0E-3 * XLI, 1.0E-3 * ZTLI, interpDF[1], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.xlim(-30.0, 50.0)
              plt.ylim(0.0, 30.0)
              plt.title('Change W - (m/s/s)')
              
              flowAngle = np.arctan(wxz[0,:] * np.reciprocal(INIT[ubdex] + uxz[0,:]))
              slopeAngle = np.arctan(dHdX)
              
              plt.subplot(2,2,2)
              plt.plot(1.0E-3 * REFS[0], flowAngle, 'b-', 1.0E-3 * REFS[0], slopeAngle, 'k--')
              plt.xlim(-20.0, 20.0)
              plt.title('Flow vector angle and terrain angle')
              
              plt.subplot(2,2,4)
              plt.plot(1.0E-3 * REFS[0], np.abs(flowAngle - slopeAngle), 'k')              
              plt.title('Boundary Constraint |Delta| - (m/s)')
              
              plt.tight_layout()
              plt.show()
              
       fig = plt.figure(figsize=(12.0, 6.0))
       # 2 X 2 subplot with all fields at the final time
       for pp in range(4):
              plt.subplot(2,2,pp+1)
              ccheck = plt.contourf(XLI, ZTLI, interpLN[pp], 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
              cbar = fig.colorbar(ccheck)
              plt.tight_layout()
       plt.show()
       
       fig = plt.figure(figsize=(12.0, 6.0))
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
              ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 201, cmap=cm.seismic)
              cbar = plt.colorbar(ccheck, format='%.3e')
              plt.tight_layout()
       plt.show()

       #%% Check the boundary conditions
       '''
       plt.figure()
       plt.plot(REFS[0],nativeLN[0][0,:])
       plt.plot(REFS[0],nativeNL[0][0,:])
       plt.title('Horizontal Velocity - Terrain Boundary')
       plt.xlim(-15000, 15000)
       plt.figure()
       plt.plot(REFS[0],nativeLN[1][0,:])
       plt.plot(REFS[0],nativeNL[1][0,:])
       plt.title('Vertical Velocity - Terrain Boundary')
       plt.xlim(-15000, 15000)
       '''
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