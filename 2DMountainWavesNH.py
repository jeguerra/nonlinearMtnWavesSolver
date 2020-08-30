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
import computeGuellrichDomain2D as coords
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeRayleighEquations import computeRayleighEquations
from computeInterpolatedFields import computeInterpolatedFields

# Numerical stuff
import computeDerivativeMatrix as derv
import computeEulerEquationsLogPLogT as eqs
import computeTimeIntegration as tint

import faulthandler; faulthandler.enable()

# Disk settings
#localDir = '/scratch/opbuffer/' # NOAA laptop
localDir = '/home/jeg/scratch/' # Home super desktop
#localDir = '/Users/TempestGuerra/scratch/' # Davis Macbook Pro
restart_file = localDir + 'restartDB'
schurName = localDir + 'SchurOps'

def displayResiduals(message, RHS, thisTime, udex, wdex, pdex, tdex):
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

def getFromRestart(name, TOPT, NX, NZ, StaticSolve):
       rdb = shelve.open(restart_file, flag='r')
       
       NX_in = rdb['NX']
       NZ_in = rdb['NZ']
       if NX_in != NX or NZ_in != NZ:
              print('ERROR: RESTART DATA IS INVALID')
              print(NX, NX_in)
              print(NZ, NZ_in)
              sys.exit(2)
       
       SOLT = rdb['SOLT']
       DCF = rdb['DCF']
       IT = rdb['ET']
       LMS = rdb['LMS']
       if TOPT[4] <= IT and not StaticSolve:
              print('ERROR: END TIME LEQ INITIAL TIME ON RESTART')
              sys.exit(2)
              
       rdb.close()
       
       return np.array(SOLT), LMS, DCF, NX_in, NZ_in, IT

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
              SB = sps.bmat([[bdb['LDIA'], bdb['LNA'], bdb['LOA']], \
                             [bdb['LDA'], bdb['A'], bdb['B']], \
                             [bdb['LHA'], bdb['E'], bdb['F']]], format='csr')
       elif blockName == 'BS':
              SB = sps.bmat([[bdb['LPA'], bdb['LQAR']], \
                             [bdb['C'], bdb['D']], \
                             [bdb['G'], bdb['H']]], format='csr')
       elif blockName == 'CS':
              SB = sps.bmat([[bdb['LMA'], bdb['I'], bdb['J']], \
                             [bdb['LQAC'], bdb['N'], bdb['O']]], format='csr')
       elif blockName == 'DS':
              SB = sps.bmat([[bdb['K'], bdb['M']], \
                             [bdb['P'], bdb['Q']]], format='csr')
       else:
              print('INVALID SCHUR BLOCK NAME!')
              
       bdb.close()

       return SB.toarray()

def runModel(TestName):
       import TestCase
       
       thisTest = TestCase.TestCase(TestName)
       
       # Need to move this to TestCase.py
       SparseDerivativesDynamics = False
       SparseDerivativesDynSGS = False
       
       # Set the solution type (MUTUALLY EXCLUSIVE)
       StaticSolve = thisTest.solType['StaticSolve']
       NonLinSolve = thisTest.solType['NLTranSolve']
       NewtonLin = thisTest.solType['NewtonLin']
       ExactBC = thisTest.solType['ExactBC']
       
       if StaticSolve:
              RSBops = False
              ApplyGML = False
       else:
              RSBops = True
              ApplyGML = True
       
       # Set the grid type
       HermCheb = thisTest.solType['HermChebGrid']
       # Use the uniform grid fourier solution if not Hermite Functions
       if not HermCheb:
              FourCheb = True
       else:
              FourCheb = False
       #FourCheb = thisTest.solType['FourChebGrid']
       
       # Set residual diffusion switch
       ResDiff = thisTest.solType['DynSGS']
       if ResDiff:
              print('DynSGS Diffusion Model.')
       else:
              print('Flow-Dependent Diffusion Model.')
       
       # Set direct solution method (MUTUALLY EXCLUSIVE)
       SolveFull = thisTest.solType['SolveFull']
       SolveSchur = thisTest.solType['SolveSchur']
       
       # Set Newton solve initial and restarting parameters
       toRestart = thisTest.solType['ToRestart'] # Saves resulting state to restart database
       isRestart = thisTest.solType['IsRestart'] # Initializes from a restart database
       makePlots = thisTest.solType['MakePlots'] # Switch for diagnostic plotting
       
       # Various background options
       smooth3Layer = thisTest.solType['Smooth3Layer']
       uniformStrat = thisTest.solType['UnifStrat']
       uniformWind = thisTest.solType['UnifWind']
       linearShear = thisTest.solType['LinShear']
       
       PHYS = thisTest.PHYS # Physical constants
       varDex = thisTest.varDex # Indeces
       DIMS = thisTest.DIMS # Grid dimensions
       JETOPS = thisTest.JETOPS # Shear jet options
       RLOPT = thisTest.RLOPT # Sponge layer options
       HOPT = thisTest.HOPT # Terrain profile options
       TOPT = thisTest.TOPT # Time integration options
       
       if RLOPT[5] == 'periodic':
              latPeriodic = True
              latInflow = False
       elif RLOPT[5] == 'inflow':
              latPeriodic = False
              latInflow = True
       elif RLOPT[5] == 'periodic_inflow':
              latPeriodic = True
              latInflow = True
       else:
              latPeriodic = False
              latInflow = True
       
       # Make the equation index vectors for all DOF
       numVar = 4
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = DIMS[5]
       udex = np.arange(OPS)
       wdex = np.add(udex, OPS)
       pdex = np.add(wdex, OPS)
       tdex = np.add(pdex, OPS)
       
       Z_in = thisTest.Z_in
       T_in = thisTest.T_in
       
       #%% SET UP THE GRID AND INITIAL STATE
       #% Define the computational and physical grids+
       REFS = computeGrid(DIMS, HermCheb, FourCheb)
      
       # Compute the raw derivative matrix operators in alpha-xi computational space
       if HermCheb and not FourCheb:
              DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       elif FourCheb and not HermCheb:
              DDX_1D, HF_TRANS = derv.computeFourierDerivativeMatrix(DIMS)
       else:
              DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
              
       DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
         
       #DDX_SP = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[0])
       #DDZ_SP = derv.computeCompactFiniteDiffDerivativeMatrix1(DIMS, REFS[1])
       
       # Update the REFS collection
       REFS.append(DDX_1D)
       REFS.append(DDZ_1D)
       
       #% Read in topography profile or compute from analytical function
       HofX, dHdX = computeTopographyOnGrid(REFS, HOPT, DDX_1D)
       
       # Compute DX and DZ grid length scales
       DX_min = 1.0 * np.min(np.abs(np.diff(REFS[0])))
       DZ_min = 1.0 * np.min(np.abs(np.diff(REFS[1])))
       print('Minimum grid lengths:',DX_min,DZ_min)
       DX_avg = 1.0 * np.mean(np.abs(np.diff(REFS[0])))
       DZ_avg = 1.0 * np.mean(np.abs(np.diff(REFS[1])))
       print('Average grid lengths:',DX_avg,DZ_avg)
       DX_max = 1.0 * np.max(np.abs(np.diff(REFS[0])))
       DZ_max = 1.0 * np.max(np.abs(np.diff(REFS[1])))
       print('Maximum grid lengths:',DX_max,DZ_max)
       DX_wav = 1.0 * abs(DIMS[1] - DIMS[0]) / (NX+1)
       DZ_wav = 1.0 * abs(DIMS[2]) / (NZ)
       print('Wavelength grid lengths:',DX_wav,DZ_wav)
              
       # Make the 2D physical domains from reference grids and topography
       zRay = DIMS[2] - RLOPT[0]
       # USE THE GUELLRICH TERRAIN DECAY
       XL, ZTL, DZT, sigma, ZRL, DXM, DZM = \
              coords.computeGuellrichDomain2D(DIMS, REFS, zRay, HofX, dHdX, StaticSolve)
       # USE UNIFORM STRETCHING
       #XL, ZTL, DZT, sigma, ZRL = coords.computeStretchedDomain2D(DIMS, REFS, zRay, HofX, dHdX)
       
       DX_local = np.reshape(DXM, (OPS,), order='F')
       DZ_local = np.reshape(DZM, (OPS,), order='F')
       
       if not StaticSolve:
              #'''
              DX = 1.0 * DX_avg
              DZ = 1.0 * DZ_avg
              DX2 = DX**2
              DZ2 = DZ**2
              #'''
              '''
              DX = 1.0 * DX_local
              DZ = 1.0 * DZ_local
              DX2 = np.power(DX, 2.0)
              DZ2 = np.power(DZ, 2.0)
              '''
              '''
              DX_spr = 1.0 / np.abs(spl.eigs(DDX_1D[1:-2,1:-2], k=1, which='LM', return_eigenvectors=False))
              DZ_spr = 1.0 / np.abs(spl.eigs(DDZ_1D[1:-2,1:-2], k=1, which='LM', return_eigenvectors=False))
              DX = 2.0 * DX_spr
              DZ = 2.0 * DZ_spr
              DX2 = DX**2
              DZ2 = DZ**2
              print('Spectral radii of 1st derivative matrices: ', DX_spr, DZ_spr)
              '''
       # Update the REFS collection
       REFS.append(XL)
       REFS.append(ZTL)
       REFS.append(dHdX)
       REFS.append(sigma)
       
       #% Compute the BC index vector
       uldex, urdex, ubdex, utdex, wbdex, \
       ubcDex, wbcDex, pbcDex, tbcDex, \
       zeroDex_tran, sysDex, diffDex = \
              computeAdjust4CBC(DIMS, numVar, varDex, latPeriodic, latInflow)
              
       # Index to interior of terrain boundary
       hdex = range(0,NX+1)
       # Index to the entire bottom boundary on U
       uBotDex = np.array(range(0, OPS, NZ))
       
       #% Read in sensible or potential temperature soundings (corner points)
       SENSIBLE = 1
       #POTENTIAL = 2
       # Map the sounding to the computational vertical 2D grid [0 H]
       TZ, DTDZ = \
              computeTemperatureProfileOnGrid(PHYS, REFS, Z_in, T_in, smooth3Layer, uniformStrat)
       '''
       # Make a figure of the temperature background
       fig = plt.figure(figsize=(12.0, 6.0))
       plt.subplot(1,2,1)
       plt.plot(T_in, 1.0E-3*np.array(Z_in), 'ko-')
       plt.title('Discrete Temperature Profile (K)')
       plt.xlabel('Temperature (K)')
       plt.ylabel('Height (km)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,2,2)
       plt.plot(TZ, 1.0E-3*ZTL, 'k-')
       plt.title('Smooth Temperature Profile (K)')
       plt.xlabel('Temperature (K)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.tight_layout()
       plt.savefig('python results/Temperature_Background.png')
       plt.show()
       # Make a figure of the temperature lapse rates background
       fig = plt.figure(figsize=(12.0, 6.0))
       plt.subplot(1,2,1)
       plt.plot(T_in, 1.0E-3*np.array(Z_in), 'ko-')
       plt.title('Discrete Temperature Profile (K)')
       plt.xlabel('Temperature (K)')
       plt.ylabel('Height (km)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,2,2)
       plt.plot(DTDZ, 1.0E-3*ZTL, 'k-')
       plt.title('Smooth Temperature Profile (K)')
       plt.xlabel('Temperature (K)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.tight_layout()
       plt.savefig('python results/Temperature_Background.png')
       plt.show()
       sys.exit(2)
       '''
       # Compute background fields on the reference column
       dlnPdz, LPZ, PZ, dlnPTdz, LPT, PT, RHO = \
              computeThermoMassFields(PHYS, DIMS, REFS, TZ[:,0], DTDZ[:,0], SENSIBLE, uniformStrat)
       
       # Read in or compute background horizontal wind profile
       U, dUdz = computeShearProfileOnGrid(REFS, JETOPS, PHYS[1], PZ, dlnPdz, uniformWind, linearShear)
       
       #% Compute the background gradients in physical 2D space
       dUdz = np.expand_dims(dUdz, axis=1)
       DUDZ = np.tile(dUdz, NX+1)
       DUDZ = computeColumnInterp(DIMS, REFS[1], dUdz, 0, ZTL, DUDZ, CH_TRANS, '1DtoTerrainFollowingCheb')
       # Compute thermodynamic gradients (no interpolation!)
       PORZ = PHYS[3] * TZ
       DLPDZ = -PHYS[0] / PHYS[3] * np.reciprocal(TZ)
       DLTDZ = np.reciprocal(TZ) * DTDZ
       DLPTDZ = DLTDZ - PHYS[4] * DLPDZ
       
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
       
       SV = PORZ * np.reciprocal(np.exp(LOGP)) # Specific volume (1 / rho_bar)
       
       #%% Rayleigh opearator and GML weight
       ROPS, RLM, GMLX, GMLZ = computeRayleighEquations(DIMS, REFS, ZRL, RLOPT, ubdex, utdex)
       if ApplyGML:
              GMLXOP = sps.diags(np.reshape(GMLX, (OPS,), order='F'), offsets=0, format='csr')
              GMLZOP = sps.diags(np.reshape(GMLZ, (OPS,), order='F'), offsets=0, format='csr')
       else:
              GMLXOP = sps.identity(OPS, format='csr')
              GMLZOP = sps.identity(OPS, format='csr')
              
       # Get the static vertical gradients and store
       DUDZ = np.reshape(DUDZ, (OPS,1), order='F')
       DLTDZ = np.reshape(DLTDZ, (OPS,1), order='F')
       DLPDZ = np.reshape(DLPDZ, (OPS,1), order='F')
       DLPTDZ = np.reshape(DLPTDZ, (OPS,1), order='F')
       DQDZ = np.hstack((DUDZ, np.zeros((OPS,1)), DLPDZ, DLPTDZ))
       
       # Make a collection for background field derivatives
       REFG = [GMLXOP, GMLZOP, DLTDZ, DQDZ, RLOPT[4], RLM]
       
       # Update the REFS collection
       REFS.append(np.reshape(UZ, (OPS,), order='F'))
       REFS.append((np.reshape(PORZ, (OPS,), order='F'), np.reshape(SV, (OPS,), order='F')))
       
       # Get some memory back here
       del(PORZ)
       del(DUDZ)
       del(DLTDZ)
       del(DLPDZ)
       del(DLPTDZ)
       del(GMLX)
       del(GMLZ)
       
       #%% Get the 2D linear operators in Hermite-Chebyshev space
       DDXM, DDZM = computePartialDerivativesXZ(DIMS, REFS, DDX_1D, DDZ_1D)
       
       #%% Get the 2D linear operators in Compact Finite Diff (for Laplacian)
       #DDXM_SP, DDZM_SP = computePartialDerivativesXZ(DIMS, REFS, DDX_SP, DDZ_SP)
       
       # For use with transient run in CSR format and RHS evaluations
       REFS.append(DDXM) # index 10
       REFS.append(DDZM) # index 11
       
       if StaticSolve or not RSBops:
              # Matrix operators for Jacobian assembly
              REFS.append(DDXM)
              REFS.append(DDZM)
       else:
              # Multithreaded enabled for transient solution
              from rsb import rsb_matrix
              REFS.append(rsb_matrix(DDXM))
              REFS.append(rsb_matrix(DDZM))
       
       # Store the terrain profilse
       REFS.append(DZT)
       DZDX = np.reshape(DZT, (OPS,1), order='F')
       DZDX2 = np.power(DZDX, 2.0)
       D2ZDX2 = DDXM.dot(DZDX)
       REFS.append(DZDX)
       REFS.append(DZDX2)
       REFS.append(D2ZDX2)
       '''
       plt.plot(REFS[0], DZDX2[ubdex])
       plt.figure()
       plt.plot(REFS[0], D2ZDX2[ubdex])
       plt.show()
       input()
       '''
       # Store the more sparse FD matrices
       #REFS.append(rsb_matrix(DDXM_SP))
       #REFS.append(rsb_matrix(DDZM_SP))
       '''
       # Check derivatives for consistency
       dcheck1 = DDXM.dot(np.ones((OPS,)))
       dcheck2 = DMX.dot(np.ones((OPS,)))
       plt.plot(dcheck1 - dcheck2, 'k-')
       plt.title('Hermite Function Derivative: (SciPy - PyRSB) Dot')
       plt.show()
       dcheck1 = DDZM.dot(np.ones((OPS,)))
       dcheck2 = DMZ.dot(np.ones((OPS,)))
       plt.plot(dcheck1 - dcheck2, 'k-')
       plt.title('Chebyshev Derivative: (SciPy - PyRSB) Dot')
       plt.show()
       input()
       '''
              
       del(DDXM)
       del(DDZM)
       del(DZDX)
       del(GMLXOP)
       del(GMLZOP)
       
       #%% SOLUTION INITIALIZATION
       physDOF = numVar * OPS
       lmsDOF = len(ubdex)
       
       # Initialize solution storage
       SOLT = np.zeros((physDOF, 2))
       
       # Initialize Lagrange Multiplier storage
       LMS = np.zeros(lmsDOF)
       
       # Initialize hydrostatic background
       INIT = np.zeros((physDOF,))
       RHS = np.zeros((physDOF,))
       
       # Initialize the Background fields
       INIT[udex] = np.reshape(UZ, (OPS,), order='F')
       INIT[wdex] = np.zeros((OPS,))
       INIT[pdex] = np.reshape(LOGP, (OPS,), order='F')
       INIT[tdex] = np.reshape(LOGT, (OPS,), order='F')
       
       # Initialize diffusion coefficients
       DCF = (np.zeros((OPS, 1)), np.zeros((OPS, 1)))
       
       if isRestart:
              print('Restarting from previous solution...')
              SOLT, LMS, DCF, NX_in, NZ_in, IT = getFromRestart(restart_file, TOPT, NX, NZ, StaticSolve)
              
              # Updates nolinear boundary condition to next Newton iteration
              dWBC = SOLT[wbdex,0] - dHdX[hdex] * (INIT[ubdex] + SOLT[ubdex,0])  
       else:
              # Set the initial time
              IT = 0.0
              
              # Initial change in vertical velocity at boundary
              dWBC = -dHdX[hdex] * INIT[ubdex]
            
       # Prepare the current fields (TO EVALUATE CURRENT JACOBIAN)
       currentState = np.array(SOLT[:,0])
       fields, U = eqs.computePrepareFields(REFS, currentState, INIT, udex, wdex, pdex, tdex)
              
       #% Compute the global LHS operator and RHS
       if StaticSolve:
              if NewtonLin:
                     # Full Newton linearization with TF terms
                     DOPS_NL = eqs.computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, \
                                   np.array(fields), U, ubdex, utdex)
              else:
                     # Classic linearization without TF terms
                     DOPS_NL = eqs.computeEulerEquationsLogPLogT_Classical(DIMS, PHYS, REFS, REFG)

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
              # Compute the RHS for this iteration
              DqDx, DqDz, DqDx_GML, DqDz_GML = \
                     eqs.computeFieldDerivatives(fields, REFS[10], REFS[11], REFG[0], REFG[1])
              rhs = eqs.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, \
                                                         REFS[15], REFS[9][0], fields, U, ubdex)
              rhs += eqs.computeRayleighTendency(REFG, fields)
              # Fix essential BC
              rhs[zeroDex_tran[0],0] *= 0.0
              rhs[zeroDex_tran[1],1] *= 0.0
              rhs[zeroDex_tran[2],2] *= 0.0
              rhs[zeroDex_tran[3],3] *= 0.0
              RHS = np.reshape(rhs, (physDOF,), order='F')
              err = displayResiduals('Current function evaluation residual: ', RHS, 0.0, udex, wdex, pdex, tdex)
              del(U); del(fields); del(rhs)
              
              #%% Compute Lagrange Multiplier column augmentation matrices (terrain equation)
              C1 = -1.0 * sps.diags(dHdX[hdex], offsets=0, format='csr')
              C2 = +1.0 * sps.eye(len(ubdex), format='csr')
       
              colShape = (OPS,len(ubdex))
              LD = sps.lil_matrix(colShape)
              if ExactBC:
                     LD[ubdex,:] = C1
              LH = sps.lil_matrix(colShape)
              LH[ubdex,:] = C2
              LM = sps.lil_matrix(colShape)
              LQ = sps.lil_matrix(colShape)
              
              #%% Apply BC adjustments and indexing block-wise (Lagrange blocks)
              LDA = LD[ubcDex,:]
              LHA = LH[wbcDex,:]
              LMA = LM[pbcDex,:]
              LQAC = LQ[tbcDex,:]
              
              # Apply transpose for row augmentation (Lagrange blocks)
              LNA = LDA.T
              LOA = LHA.T
              LPA = LMA.T
              LQAR = LQAC.T
              LDIA = sps.lil_matrix((lmsDOF,lmsDOF))
              
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
              if (StaticSolve and SolveSchur):
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
                     opdb['LDA'] = LDA; opdb['LHA'] = LHA; opdb['LMA'] = LMA; opdb['LQAC'] = LQAC
                     opdb['LNA'] = LNA; opdb['LOA'] = LOA; opdb['LPA'] = LPA; opdb['LQAR'] = LQAR
                     opdb['LDIA'] = LDIA
                     opdb.close()
                      
                     # Compute the partitions for Schur Complement solution
                     fu = RHS[udex]
                     fw = RHS[wdex]
                     f1 = np.concatenate((-dWBC, fu[ubcDex], fw[wbcDex]))
                     fp = RHS[pdex]
                     ft = RHS[tdex]
                     f2 = np.concatenate((fp[pbcDex], ft[tbcDex]))
                     
              if (StaticSolve and SolveFull):
                     # Add Rayleigh damping terms
                     A += R1
                     F += R2
                     K += R3
                     Q += R4
                     
                     # Compute the global linear operator
                     AN = sps.bmat([[A, B, C, D, LDA], \
                              [E, F, G, H, LHA], \
                              [I, J, K, M, LMA], \
                              [N, O, P, Q, LQAC], \
                              [LNA, LOA, LPA, LQAR, LDIA]], format='csc')
              
                     # Compute the global linear force vector
                     bN = np.concatenate((RHS[sysDex], -dWBC))
              
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
                     opts = dict(Equil=True, IterRefine='DOUBLE')
                     factor = spl.splu(AN, permc_spec='MMD_ATA', options=opts)
                     del(AN)
                     dsol = factor.solve(bN)
                     del(bN)
                     del(factor)
              if SolveSchur and not SolveFull:
                     print('Solving linear system by Schur Complement...')
                     # Factor DS and compute the Schur Complement of DS
                     DS = computeSchurBlock(schurName,'DS')
                     factorDS = dsl.lu_factor(DS, overwrite_a=True)
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
                     factorDS_SC = dsl.lu_factor(AS, overwrite_a=True)
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
              # Store the Lagrange Multipliers
              LMS += dsol[0:lmsDOF]
              dsolQ = dsol[lmsDOF:]
              
              SOLT[sysDex,0] += dsolQ

              # Store solution change to instance 1
              SOLT[sysDex,1] = dsolQ
              
              print('Recover full linear solution vector... DONE!')
              
              # Prepare the fields
              fields, U = eqs.computePrepareFields(REFS, np.array(SOLT[:,0]), INIT, udex, wdex, pdex, tdex)
              
              #%% Set the output residual and check
              message = 'Residual 2-norm BEFORE Newton step:'
              err = displayResiduals(message, RHS, 0.0, udex, wdex, pdex, tdex)
              DqDx, DqDz, DqDx_GML, DqDz_GML = \
                     eqs.computeFieldDerivatives(fields, REFS[10], REFS[11], REFG[0], REFG[1])
              rhs = eqs.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, \
                                                         REFS[15], REFS[9][0], fields, U, ubdex)
              rhs += eqs.computeRayleighTendency(REFG, fields)
              # Fix essential BC
              rhs[zeroDex_tran[0],0] *= 0.0
              rhs[zeroDex_tran[1],1] *= 0.0
              rhs[zeroDex_tran[2],2] *= 0.0
              rhs[zeroDex_tran[3],3] *= 0.0
              RHS = np.reshape(rhs, (physDOF,), order='F')
              message = 'Residual 2-norm AFTER Newton step:'
              err = displayResiduals(message, RHS, 0.0, udex, wdex, pdex, tdex)
              del(rhs)
              
              # Check the change in the solution
              DSOL = np.array(SOLT[:,1])
              print('Norm of change in solution: ', np.linalg.norm(DSOL))
       #%% Transient solutions       
       elif NonLinSolve:
              print('Starting Nonlinear Transient Solver...')
                            
              # Reshape main solution vectors and initialize
              hydroState = np.reshape(INIT, (OPS, numVar), order='F')
              rhsVec = np.zeros((OPS, numVar))
              error = [np.linalg.norm(rhsVec)]
              
              # Initialize boundary solution
              fields[ubdex,1] = -dWBC
                            
              # Initialize time constants
              ti = 0
              ff = 1
              thisTime = IT
              
              # Initialize netcdf output
              from netCDF4 import Dataset
              fname = 'transientNL' + str(int(thisTime)) + '.nc'
              try:
                     m_fid = Dataset(fname, 'w', format="NETCDF4")
              except PermissionError:
                     print('Deleting corrupt NC file... from failed run.')
                     import os
                     os.remove(fname)
                     m_fid = Dataset(fname, 'w', format="NETCDF4")
                     
              # Make dimensions
              taxis = m_fid.createDimension('time', None)
              xaxis = m_fid.createDimension('xlon', NX+1)
              zaxis = m_fid.createDimension('zlev', NZ)
              # Create variables (time and grid)
              tmvar = m_fid.createVariable('t', 'f8', ('time',))
              xgvar = m_fid.createVariable('XL', 'f8', ('zlev', 'xlon'))
              zgvar = m_fid.createVariable('ZTL', 'f8', ('zlev', 'xlon'))
              # Store variables
              xgvar[:] = XL
              zgvar[:] = ZTL
              # Create variables (background static fields)
              UVAR = m_fid.createVariable('U', 'f8', ('zlev', 'xlon'))
              PVAR = m_fid.createVariable('LNP', 'f8', ('zlev', 'xlon'))
              TVAR = m_fid.createVariable('LNT', 'f8', ('zlev', 'xlon'))
              # Store variables
              UVAR[:] = np.reshape(hydroState[:,0], (NZ,NX+1), order='F')
              PVAR[:] = np.reshape(hydroState[:,2], (NZ,NX+1), order='F')
              TVAR[:] = np.reshape(hydroState[:,3], (NZ,NX+1), order='F')
              # Create variables (transient fields)
              uvar = m_fid.createVariable('u', 'f8', ('time', 'zlev', 'xlon'))
              wvar = m_fid.createVariable('w', 'f8', ('time', 'zlev', 'xlon'))
              pvar = m_fid.createVariable('ln_p', 'f8', ('time', 'zlev', 'xlon'))
              tvar = m_fid.createVariable('ln_t', 'f8', ('time', 'zlev', 'xlon'))
              
              # Initialize local sound speed and time step
              VSND = np.sqrt(PHYS[6] * REFS[9][0])
              VSND_max = np.amax(VSND)
              DT0 = min(DX_min / np.amax(VSND), DZ_min / np.amax(VSND))
              TOPT[0] = DT0
              print('Initial time step: ', str(DT0) + ' (sec)')
              
              OTI = int(TOPT[5] / DT0)
              ITI = int(TOPT[6] / DT0)
              
              import bottleneck as bn
              import computeResidualViscCoeffs as rescf
              import computeTimeIntegrationDynamics as dynint
              import computeTimeIntegrationDiffusion as difint
              
              while thisTime <= TOPT[4]:
                            
                     # Print out diagnostics every TOPT[5] steps
                     if ti % OTI == 0:
                            message = ''
                            err = displayResiduals(message, np.reshape(rhsVec, (OPS*numVar,), order='F'), thisTime, udex, wdex, pdex, tdex)
                            error.append(err)
                            
                            # Compute the maximum speed of sound
                            T_ratio = np.exp(PHYS[4] * fields[:,2] + fields[:,3]) - 1.0
                            gamRdT = PHYS[6] * REFS[9][0] * (1.0 + T_ratio)
                            VSND_max = np.amax(np.sqrt(gamRdT))
                            DTN = min(DX_min / np.amax(VSND_max), DZ_min / np.amax(VSND_max))
                            DT_factor = DTN / DT0
                            TOPT[0] *= DT_factor
                            DT0 = DTN
                            #print('Maximum sound speed (m/s): ', VSND_max)
                            #print('NEW time step estimate (CFL = 1): ', TOPT[0])
                     
                     if ti % ITI == 0:
                            # Store current time
                            tmvar[ff-1] = thisTime
                            fig = plt.figure(figsize=(8.0, 12.0))
                            # Check the fields or tendencies
                            for pp in range(numVar):
                                   plt.subplot(4,1,pp+1)
                                   dqdt = np.reshape(fields[:,pp], (NZ, NX+1), order='F')
                                   
                                   if np.abs(dqdt.max()) > np.abs(dqdt.min()):
                                          clim = np.abs(dqdt.max())
                                   elif np.abs(dqdt.max()) < np.abs(dqdt.min()):
                                          clim = np.abs(dqdt.min())
                                   else:
                                          clim = np.abs(dqdt.max())
                                  
                                   ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 101, cmap=cm.seismic, vmin=-clim, vmax=clim)
                                   plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
                                   #plt.xlim(-30.0, 30.0)
                                   #plt.ylim(0.0, 20.0)
                                   
                                   if pp < (numVar - 1):
                                          plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                                   else:
                                          plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                                          
                                   plt.colorbar(ccheck, format='%.2e')
                                   
                                   if pp == 0:
                                          plt.title('u (m/s): ' + 'Time = {:.5f} (sec)'.format(thisTime))
                                          uvar[ff-1,:,:] = dqdt
                                   elif pp == 1:
                                          plt.title('w (m/s): ' + 'dT = {:.5f} (sec)'.format(TOPT[0]))
                                          wvar[ff-1,:,:] = dqdt
                                   elif pp == 2:
                                          plt.title('ln-p (Pa)')
                                          pvar[ff-1,:,:] = dqdt
                                   else:
                                          plt.title('ln-theta (K)')
                                          tvar[ff-1,:,:] = dqdt
       
                            plt.show()
                            ff += 1

                     if ti == 0:
                            isFirstStep = True
                            UD = np.abs(fields[:,0] + hydroState[:,0])
                            WD = np.abs(fields[:,1])
                            DCF = rescf.computeFlowVelocityCoeffs(UD, WD, DX, DZ)
                     else:
                            isFirstStep = False
                            
                     # Compute the solution within a time step
                     #'''
                     fields_new, rhsVec_new, thisTime = tint.computeTimeIntegrationNL2(PHYS, REFS, REFG, \
                                                             DX, DZ, DX2, DZ2,\
                                                             TOPT, fields, hydroState, DCF, \
                                                             zeroDex_tran, diffDex, \
                                                             ResDiff, thisTime, isFirstStep)
                     
                     # Compute residual and normalizations
                     WD = np.abs(fields_new[:,1])
                     # Compute DynSGS or Flow Dependent diffusion coefficients
                     if ResDiff:
                            UD = np.abs(fields_new[:,0] + hydroState[:,0])
                            QM = bn.nanmax(fields_new, axis=0)
                            # Trapezoidal Rule estimate of residual
                            resInv = (1.0 / TOPT[0]) * (fields_new - fields) - 0.5 * (rhsVec_new + rhsVec)
                            
                            DCF = rescf.computeResidualViscCoeffs(resInv, QM, UD, WD, DX, DZ, DX**2, DZ**2, REFG[5])
                     else:
                            UD = np.abs(fields_new[:,0] + hydroState[:,0])
                            DCF = rescf.computeFlowVelocityCoeffs(UD, WD, DX, DZ)
                            
                     fields = np.array(fields_new)
                     rhsVec = np.array(rhsVec_new)
                     del(fields_new)
                     del(rhsVec_new)
                     #'''
                     '''
                     #%% Time integration by applying diffusion AFTER dynamics update
                     # Dynamics step
                     fields_new, rhsVec_new = dynint.computeTimeIntegrationDYNCS(PHYS, REFS, REFG, \
                                                             TOPT, fields, hydroState, \
                                                             zeroDex_tran, ubdex, isFirstStep)
                            
                     # Compute residual and normalizations
                     UD = np.abs(fields_new[:,0] + hydroState[:,0])
                     WD = np.abs(fields_new[:,1])
                     
                     # Compute DynSGS or Flow Dependent diffusion coefficients
                     if ResDiff:
                            QM = bn.nanmax(fields_new, axis=0)
                            if isFirstStep:
                                   # Backward Euler estimate of residual
                                   resInv = (1.0 / TOPT[0]) * (fields_new - fields) - rhsVec_new
                            else:
                                   # Trapezoidal Rule estimate of residual
                                   resInv = (1.0 / TOPT[0]) * (fields_new - fields) - 0.5 * (rhsVec_new + rhsVec)
                            dcoeff = rescf.computeResidualViscCoeffs(resInv, QM, UD, WD, DX_local, DZ_local, REFG[5])
                     else:
                            dcoeff = rescf.computeFlowVelocityCoeffs(UD, WD, DX_local, DZ_local)
                            
                     # Diffusion step
                     fields_new = difint.computeTimeIntegrationVISC(PHYS, REFS, REFG, \
                                                             TOPT, fields_new, hydroState, dcoeff, \
                                                             zeroDex_tran, ubdex, \
                                                             (extDex, latDex, vrtDex), isFirstStep)
                     fields = np.array(fields_new)
                     del(fields_new)
                     del(rhsVec_new)
                     
                     # Compute first derivatives
                     if isFirstStep:
                            DDXM = REFS[10]
                            DDZM = REFS[11]
                     else:
                            DDXM = REFS[12]
                            DDZM = REFS[13]
                         
                     U = fields[:,0] + hydroState[:,0]
                     DqDx, DqDz, DqDx_GML, DqDz_GML = \
                            eqs.computeFieldDerivatives(fields, DDXM, DDZM, REFG[0], REFG[1])
                     # Compute dynamical tendencies
                     rhsVec = eqs.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, REFS[15], REFS[9], fields, U)
                     rhsVec += eqs.computeRayleighTendency(REFG, fields)
                     
                     # Fix Essential boundary conditions
                     rhsVec[zeroDex_tran[0],0] *= 0.0
                     rhsVec[zeroDex_tran[1],1] *= 0.0
                     rhsVec[zeroDex_tran[2],2] *= 0.0
                     rhsVec[zeroDex_tran[3],3] *= 0.0
                     thisTime += TOPT[0]
                     '''
                     ti += 1
                     
              # Close the output data file
              m_fid.close()
                     
              # Reshape back to a column vector after time loop
              SOLT[:,0] = np.reshape(fields, (OPS*numVar, ), order='F')
              RHS = np.reshape(rhsVec, (OPS*numVar, ), order='F')
              
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
              rdb['DCF'] = DCF
              rdb['NX'] = NX
              rdb['NZ'] = NZ
              rdb['ET'] = TOPT[4]
              rdb['PHYS'] = PHYS
              rdb['DIMS'] = DIMS
              if StaticSolve:
                     rdb['REFS'] = REFS
              rdb.close()
       
       #%% Recover the solution (or check the residual)
       uxz = np.reshape(SOLT[udex,0], (NZ,NX+1), order='F') 
       wxz = np.reshape(SOLT[wdex,0], (NZ,NX+1), order='F')
       pxz = np.reshape(SOLT[pdex,0], (NZ,NX+1), order='F') 
       txz = np.reshape(SOLT[tdex,0], (NZ,NX+1), order='F')
       
       #%% Make some plots for static or transient solutions
       if makePlots:
              if StaticSolve:
                     fig = plt.figure(figsize=(12.0, 6.0))
                     # 1 X 3 subplot of W for linear, nonlinear, and difference
                     
                     plt.subplot(2,2,1)
                     ccheck = plt.contourf(1.0E-3 * XL, 1.0E-3 * ZTL, uxz, 101, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
                     fig.colorbar(ccheck)
                     plt.xlim(-30.0, 50.0)
                     plt.ylim(0.0, 1.0E-3*DIMS[2])
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                     plt.title('Change U - (m/s)')
                     
                     plt.subplot(2,2,3)
                     ccheck = plt.contourf(1.0E-3 * XL, 1.0E-3 * ZTL, wxz, 101, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
                     fig.colorbar(ccheck)
                     plt.xlim(-30.0, 50.0)
                     plt.ylim(0.0, 1.0E-3*DIMS[2])
                     plt.title('Change W - (m/s)')
                     
                     flowAngle = np.arctan(wxz[0,:] * np.reciprocal(INIT[uBotDex] + uxz[0,:]))
                     slopeAngle = np.arctan(dHdX)
                     
                     plt.subplot(2,2,2)
                     plt.plot(1.0E-3 * REFS[0], flowAngle, 'b-', 1.0E-3 * REFS[0], slopeAngle, 'k--')
                     plt.xlim(-20.0, 20.0)
                     plt.title('Flow vector angle and terrain angle')
                     
                     plt.subplot(2,2,4)
                     plt.plot(1.0E-3 * REFS[0], np.abs(flowAngle - slopeAngle), 'k')              
                     plt.title('Boundary Constraint |Delta| - (m/s)')
                     
                     plt.tight_layout()
                     #plt.savefig('IterDelta_BoundaryCondition.png')
                     plt.show()
                     
              fig = plt.figure(figsize=(12.0, 6.0))
              # 2 X 2 subplot with all fields at the final time
              for pp in range(4):
                     plt.subplot(2,2,pp+1)
                     
                     if pp == 0:
                            ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, uxz, 51, cmap=cm.seismic)#, vmin=-0.25, vmax=0.25)
                            plt.title('U (m/s)')
                            plt.ylabel('Height (km)')
                     elif pp == 1:
                            ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, wxz, 51, cmap=cm.seismic)#, vmin=-0.08, vmax=0.08)
                            plt.title('W (m/s)')
                     elif pp == 2:
                            ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, pxz, 51, cmap=cm.seismic)#, vmin=-4.5E-5, vmax=4.5E-5)
                            plt.title('log-P (Pa)')
                            plt.xlabel('Distance (km)')
                            plt.ylabel('Height (km)')
                     elif pp == 3:
                            ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, txz, 51, cmap=cm.seismic)#, vmin=-6.0E-4, vmax=6.0E-4)
                            plt.title('log-Theta (K)')
                            plt.xlabel('Distance (km)')
                            
                     fig.colorbar(ccheck, format='%.3E')
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                     plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
              plt.tight_layout()
              #plt.savefig('SolutionFields.png')
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
                     plt.colorbar(ccheck, format='%+.3E')
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                     plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
                     plt.tight_layout()
              plt.show()
              
              # Check W at the boundaries...
              dwdt = np.reshape(RHS[wdex], (NZ, NX+1), order='F')
              return (XL, ZTL, dwdt)
       
if __name__ == '__main__':
       
       #TestName = 'ClassicalSchar01'
       TestName = 'ClassicalScharIter'
       #TestName = 'SmoothStratScharIter'
       #TestName = 'DiscreteStratScharIter'
       #TestName = 'CustomTest'
       
       # Run the model in a loop if needed...
       for ii in range(1):
              diagOutput = runModel(TestName)