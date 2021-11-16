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
import bottleneck as bn
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import scipy.linalg as dsl
from matplotlib import cm
import matplotlib.pyplot as plt
# Import from the local library of routines
from computeGrid import computeGrid
from computeAdjust4CBC import computeAdjust4CBC
from computeColumnInterp import computeColumnInterp
import computePartialDerivativesXZ as devop
from computeTopographyOnGrid import computeTopographyOnGrid
import computeGuellrichDomain2D as coords
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeRayleighEquations import computeRayleighEquations

# Numerical stuff
import computeDerivativeMatrix as derv
import computeEulerEquationsLogPLogT as eqs
import computeTimeIntegration as tint

import faulthandler; faulthandler.enable()

from netCDF4 import Dataset

# Disk settings
#localDir = '/scratch/opbuffer/' # NOAA laptop
localDir = '/home/jeg/scratch/' # Home super desktop
#localDir = '/Users/TempestGuerra/scratch/' # Davis Macbook Pro
#localDir = '/home/jeguerra/scratch/'
restart_file = localDir + 'restartDB'
schurName = localDir + 'SchurOps'

#import pnumpy as pnp
#pnp.enable()

def makeTemperatureBackgroundPlots(Z_in, T_in, ZTL, TZ, DTDZ):
       
       # Make a figure of the temperature background
       plt.figure(figsize=(12.0, 6.0))
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
       plt.figure(figsize=(12.0, 6.0))
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

       return       

def makeFieldPlots(TOPT, thisTime, XL, ZTL, fields, rhs, res, NX, NZ, numVar):
       fig = plt.figure(num=1, clear=True, figsize=(26.0, 18.0)) 
       for pp in range(numVar):
              q = np.reshape(fields[:,pp], (NZ+1, NX+1), order='F')
              dqdt = np.reshape(rhs[:,pp], (NZ+1, NX+1), order='F')
              residual = np.reshape(res[:,pp], (NZ+1, NX+1), order='F')
              rowDex = 1 + 3*pp
              
              #%% Plot of the full field
              plt.subplot(4,3,rowDex)
              if np.abs(q.max()) > np.abs(q.min()):
                     clim = np.abs(q.max())
              elif np.abs(q.max()) < np.abs(q.min()):
                     clim = np.abs(q.min())
              else:
                     clim = np.abs(q.max())
             
              ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, q, 101, cmap=cm.seismic, vmin=-clim, vmax=clim)
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
              elif pp == 1:
                     plt.title('w (m/s): ' + 'dT = {:.5f} (sec)'.format(TOPT[0]))
              elif pp == 2:
                     plt.title('(ln$p$)\' (Pa)')
              else:
                     plt.title(r'(ln$\theta$)' + '\' (K)')
                     
              #%% Plot the full tendencies
              plt.subplot(4,3,rowDex + 1)
              if np.abs(dqdt.max()) > np.abs(dqdt.min()):
                     clim = np.abs(dqdt.max())
              elif np.abs(dqdt.max()) < np.abs(dqdt.min()):
                     clim = np.abs(dqdt.min())
              else:
                     clim = np.abs(dqdt.max())
             
              ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 101, cmap=cm.seismic, vmin=-clim, vmax=clim)
              plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
              if pp < (numVar - 1):
                     plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
              else:
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                     
              plt.colorbar(ccheck, format='%.2e')
              
              if pp == 0:
                     plt.title('du/dt (m/s2): ' + 'Time = {:.5f} (sec)'.format(thisTime))
              elif pp == 1:
                     plt.title('dw/dt (m/s2): ' + 'dT = {:.5f} (sec)'.format(TOPT[0]))
              elif pp == 2:
                     plt.title('d(ln$p$)\'' + '/dt' + ' (Pa/s)')
              else:
                     plt.title(r'd(ln$\theta$)' + '\'/dt' + ' (K/s)')
                     
              #plt.xlim(-50.0, 50.0)
              #plt.ylim(0.0, 25.0)
                     
              #%% Plot the full residuals
              plt.subplot(4,3,rowDex + 2)
              if np.abs(residual.max()) > np.abs(residual.min()):
                     clim = np.abs(residual.max())
              elif np.abs(residual.max()) < np.abs(residual.min()):
                     clim = np.abs(residual.min())
              else:
                     clim = np.abs(residual.max())
              '''
              if clim > 0.0:
                     normr = 1.0 / clim
              else:
                     normr = 1.0
              '''       
              normr = 1.0
              ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, normr * residual, 101, cmap=cm.seismic, vmin=-clim, vmax=clim)
              plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
              if pp < (numVar - 1):
                     plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
              else:
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
                     
              plt.colorbar(ccheck, format='%.2e')
              
              if pp == 0:
                     plt.title('Residual: u')
              elif pp == 1:
                     plt.title('Residual: w')
              elif pp == 2:
                     plt.title('Residual: ' + '(ln$p$)\'')
              else:
                     plt.title('Residual: ' + r'(ln$\theta$)' + '\'')
                     
              #plt.xlim(-50.0, 50.0)
              #plt.ylim(0.0, 25.0)
                     
       plt.tight_layout()
       #plt.show(block=False)
       plt.savefig('/media/jeg/FastDATA/linearMtnWavesSolver/animations/transient' + '{:0>6d}'.format(int(thisTime)) +  '.pdf')
       fig.clear()
       del(fig)
       
       return

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
       NCPU = int(1.25 * mtp.cpu_count())
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

def initializeNetCDF(thisTime, NX, NZ, XL, ZTL, hydroState):
       # Rename output file to the current time for subsequent storage
       fname = 'transientNL' + str(int(thisTime)) + '.nc'
       try:
              m_fid = Dataset(fname, 'w', format="NETCDF4")
       except PermissionError:
              print('Deleting corrupt NC file... from failed run.')
              import os
              os.remove(fname)
              m_fid = Dataset(fname, 'w', format="NETCDF4")
              
       # Make dimensions
       m_fid.createDimension('time', None)
       m_fid.createDimension('xlon', NX+1)
       m_fid.createDimension('zlev', NZ+1)
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
       UVAR[:] = np.reshape(hydroState[:,0], (NZ+1,NX+1), order='F')
       PVAR[:] = np.reshape(hydroState[:,2], (NZ+1,NX+1), order='F')
       TVAR[:] = np.reshape(hydroState[:,3], (NZ+1,NX+1), order='F')
       # Create variables (transient fields)
       uvar = m_fid.createVariable('u', 'f8', ('time', 'zlev', 'xlon'))
       wvar = m_fid.createVariable('w', 'f8', ('time', 'zlev', 'xlon'))
       pvar = m_fid.createVariable('ln_p', 'f8', ('time', 'zlev', 'xlon'))
       tvar = m_fid.createVariable('ln_t', 'f8', ('time', 'zlev', 'xlon'))
       # Create variables (field tendencies)
       duvar = m_fid.createVariable('DuDt', 'f8', ('time', 'zlev', 'xlon'))
       dwvar = m_fid.createVariable('DwDt', 'f8', ('time', 'zlev', 'xlon'))
       dpvar = m_fid.createVariable('Dln_pDt', 'f8', ('time', 'zlev', 'xlon'))
       dtvar = m_fid.createVariable('Dln_tDt', 'f8', ('time', 'zlev', 'xlon'))
       # Create variables (diffusion coefficients)
       dvar0 = m_fid.createVariable('CRES_X', 'f8', ('time', 'zlev', 'xlon'))
       dvar1 = m_fid.createVariable('CRES_Z', 'f8', ('time', 'zlev', 'xlon'))
       
       return m_fid, tmvar, uvar, wvar, pvar, tvar, duvar, dwvar, dpvar, dtvar, dvar0, dvar1

def runModel(TestName):
       import TestCase
       
       thisTest = TestCase.TestCase(TestName)
       
       # Set the solution type (MUTUALLY EXCLUSIVE)
       StaticSolve = thisTest.solType['StaticSolve']
       NonLinSolve = thisTest.solType['NLTranSolve']
       NewtonLin = thisTest.solType['NewtonLin']
       ExactBC = thisTest.solType['ExactBC']
       
       # Switch to use the PyRSB multithreading module (CPU multithreaded SpMV)
       if StaticSolve:
              RSBops = False
              ApplyGML = False
       else:
              RSBops = True # Turn off PyRSB SpMV
              ApplyGML = True
       
       # Set the grid type
       HermCheb = thisTest.solType['HermChebGrid']
       # Use the uniform grid fourier solution if not Hermite Functions
       if not HermCheb:
              FourCheb = True
       else:
              FourCheb = False
              
       # Set residual diffusion switch
       DynSGS = thisTest.solType['DynSGS']
       if DynSGS:
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
       
       if RLOPT[5] == 'uwpt_static':
              bcType = 1
              print('BC profile: ' + RLOPT[5])
       elif RLOPT[5] == 'uwpt_transient':
              bcType = 2
              print('BC profile: ' + RLOPT[5])
       else:
              bcType = 1
              print('DEFAULT BC profile: ' + RLOPT[5])
       
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
       
       #%% Define the computational and physical grids+
       verticalChebGrid = False
       verticalLegdGrid = True
       verticalLagrGrid = False
       
       if verticalChebGrid:
              interpolationType = '1DtoTerrainFollowingCheb'
              
       if verticalLegdGrid:
              interpolationType = '1DtoTerrainFollowingLegr'
       
       #%% FLAGS FOR SEM TESTING...
       isSEM_X = False
       isSEM_Z = False
       isNonCoincident = True
       minf = 6
       CORDER = 10
       if isSEM_X and isSEM_Z:
              NEZ = 32 #int(np.ceil(NZ / 5))
              NEX = 36 #int(np.ceil(NX / 5))
              DIMS[3] = int(2 * np.ceil(DIMS[3] / NEX / 2))
              DIMS[4] = int(2 * np.ceil(DIMS[4] / NEZ / 2)) #2
              HermCheb = True
              FourCheb = False
              xnf = max(minf, int(NEX/3) - 1)
              znf = max(minf, int(NEZ/3) - 1)
              NE = (znf, xnf)
       elif isSEM_X and not isSEM_Z:
              NEX = int(np.ceil(NX / 2))
              DIMS[3] = 2
              HermCheb = True
              FourCheb = False
              xnf = max(minf, int(NEX/4) + 1)
              NE = (minf, xnf)
       elif not isSEM_X and isSEM_Z:
              NEZ = int(np.ceil(NZ / 5))
              DIMS[4] = 5
              HermCheb = True
              FourCheb = False
              znf = max(minf, int(NEZ/4) + 1)
              NE = (znf, minf)
       else:
              NE = (minf,minf)
              
       print('DynSGS filter size: ', NE)
       if isSEM_X:
              print('@@ USING ' + str(DIMS[3]) + ' ELEMENTS IN X @@')
              print('@@ INTERIOR ORDER: ' + str(NEX) + ' @@')
              
       if isSEM_Z:
              print('@@ USING ' + str(DIMS[4]) + ' ELEMENTS IN Z @@')
              print('@@ INTERIOR ORDER: ' + str(NEZ) + ' @@')
       
       #%% COMPUTE STRATIFICATION AT HIGH RESOLUTION SPECTRAL
       DIM0 = [DIMS[0], DIMS[1], DIMS[2], DIMS[3], 256, DIMS[5]]
       REF0 = computeGrid(DIM0, HermCheb, FourCheb, verticalChebGrid, verticalLegdGrid, verticalLagrGrid)
       
       if verticalChebGrid:
              DDZP, CHT = derv.computeChebyshevDerivativeMatrix(DIM0)
              
       if verticalLegdGrid:
              DDZP, CHT = derv.computeLegendreDerivativeMatrix(DIM0)
       
       if isSEM_X:
              DDXP, REF0[0] = derv.computeSpectralElementDerivativeMatrix(REF0[0], NEX, isNonCoincident, (False, False), CORDER)
              DIM0[3] = len(REF0[0]) - 1
       else:
              DDXP, HHT = derv.computeHermiteFunctionDerivativeMatrix(DIM0)
                     
       REF0.append(DDXP)
       REF0.append(DDZP)
       
       hx, dhx = computeTopographyOnGrid(REF0, HOPT, DDXP)
       zRay = DIMS[2] - RLOPT[0]
       xl, ztl, dzt, sig, ZRL, DXM, DZM = \
              coords.computeGuellrichDomain2D(DIM0, REF0, zRay, hx, dhx, StaticSolve)
       
       REF0.append(xl)
       REF0.append(ztl)
       REF0.append(dhx)
       REF0.append(sig)
       
       tz, dtz, dtz2 = \
              computeTemperatureProfileOnGrid(PHYS, REF0, Z_in, T_in, smooth3Layer, uniformStrat)
              
       dlpz, lpz, pz, dlptz, lpt, pt, rho = \
              computeThermoMassFields(PHYS, DIM0, REF0, tz[:,0], dtz[:,0], 'sensible', uniformStrat)
              
       uz, duz = computeShearProfileOnGrid(REF0, JETOPS, PHYS[1], pz, dlpz, uniformWind, linearShear)
       
       #%% SET UP THE GRID AND INDEX VECTORS
       REFS = computeGrid(DIMS, HermCheb, FourCheb, verticalChebGrid, verticalLegdGrid, verticalLagrGrid)
      
       #%% Compute the raw derivative matrix operators in alpha-xi computational space
       if isSEM_X:
              domx = REFS[0]
              DDX_1D, REFS[0] = derv.computeSpectralElementDerivativeMatrix(domx, NEX, isNonCoincident, (False, False), CORDER)
              DIMS[3] = len(REFS[0]) - 1
              NX = DIMS[3]
              OPS = (NX + 1) * (NZ + 1)
              DIMS[5] = OPS
              udex = np.arange(OPS)
              wdex = np.add(udex, OPS)
              pdex = np.add(wdex, OPS)
              tdex = np.add(pdex, OPS)
       else:
              if HermCheb and not FourCheb:
                     DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
              elif FourCheb and not HermCheb:
                     DDX_1D, HF_TRANS = derv.computeFourierDerivativeMatrix(DIMS)
              else:
                     DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
                            
       if isSEM_Z:
              domz = np.copy(REFS[1])
              DDZ_1D, REFS[1] = derv.computeSpectralElementDerivativeMatrix(domz, NEZ, isNonCoincident, (False, False), CORDER)
              DIMS[4] = len(REFS[1]) - 1
              NZ = DIMS[4]
              OPS = (NX + 1) * (NZ + 1)
              DIMS[5] = OPS
              udex = np.arange(OPS)
              wdex = np.add(udex, OPS)
              pdex = np.add(wdex, OPS)
              tdex = np.add(pdex, OPS)
       else:
              if verticalChebGrid:
                     DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
              
              if verticalLegdGrid:
                     DDZ_1D, CH_TRANS = derv.computeLegendreDerivativeMatrix(DIMS)
       
       #%% Set derivative operators for diffusion
       if isSEM_X and isSEM_Z:
              DDX_CS = np.copy(DDX_1D)
              DDZ_CS = np.copy(DDZ_1D)
              DDX2_CS = DDX_CS.dot(DDX_CS)
              DDZ2_CS = DDZ_CS.dot(DDZ_CS)
       elif isSEM_X and not isSEM_Z:
              DDX_CS = np.copy(DDX_1D)
              DDX2_CS = DDX_CS.dot(DDX_CS)
              DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(REFS[1], True, False, False, False, DDZ_1D)
       elif not isSEM_X and isSEM_Z:
              DDX_CS, DDX2_CS = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, False, False, DDX_1D)
              DDZ_CS = np.copy(DDZ_1D)
              DDZ2_CS = DDZ_CS.dot(DDZ_CS)
       else:
              DDX_CS, DDX2_CS = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, False, False, DDX_1D)
              DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(REFS[1], True, False, False, False, DDZ_1D)
              
       #%% Update the REFS collection
       REFS.append(DDX_1D) # index 2
       REFS.append(DDZ_1D) # index 3
       
       #% Read in topography profile or compute from analytical function
       HofX, dHdX = computeTopographyOnGrid(REFS, HOPT, DDX_1D)
              
       # Make the 2D physical domains from reference grids and topography
       zRay = DIMS[2] - RLOPT[0]
       # USE THE GUELLRICH TERRAIN DECAY
       XL, ZTL, DZT, sigma, ZRL, DXM, DZM = \
              coords.computeGuellrichDomain2D(DIMS, REFS, zRay, HofX, dHdX, StaticSolve)
       # USE UNIFORM STRETCHING
       #XL, ZTL, DZT, sigma, ZRL = coords.computeStretchedDomain2D(DIMS, REFS, zRay, HofX, dHdX)
       
       # Compute DX and DZ grid length scales
       DX_min = 1.0 * np.min(np.abs(DXM))
       DZ_min = 1.0 * np.min(np.abs(DZM))
       print('Minimum grid lengths:',DX_min,DZ_min)
       DX_avg = 1.0 * np.mean(np.abs(DXM))
       DZ_avg = 1.0 * np.mean(np.abs(DZM))
       print('Average grid lengths:',DX_avg,DZ_avg)
       DX_max = 1.0 * np.max(np.abs(DXM))
       DZ_max = 1.0 * np.max(np.abs(DZM))
       print('Maximum grid lengths:',DX_max,DZ_max)
       DX_wav = 1.0 * abs(DIMS[1] - DIMS[0]) / (NX+1)
       DZ_wav = 1.0 * abs(DIMS[2]) / (NZ)
       print('Wavelength grid lengths:',DX_wav,DZ_wav)
       
       # Update the REFS collection
       REFS.append(XL)
       REFS.append(ZTL)
       REFS.append(dHdX)
       REFS.append(sigma)
       
       #% Compute the BC index vector
       uldex, urdex, ubdex, utdex, wbdex, \
       ubcDex, wbcDex, pbcDex, tbcDex, \
       zeroDex, sysDex, ebcDex = \
              computeAdjust4CBC(DIMS, numVar, varDex, bcType)
              
       # Index to interior of terrain boundary
       hdex = range(0,NX+1)
       # Index to the entire bottom boundary on U
       uBotDex = np.array(range(0, OPS, NZ+1))
       
       #%% MAKE THE INITIAL/BACKGROUND STATE ON COMPUTATIONAL GRID
       # Map the sounding to the computational vertical 2D grid [0 H]
       TZ, DTDZ, D2TDZ2 = \
              computeTemperatureProfileOnGrid(PHYS, REFS, Z_in, T_in, smooth3Layer, uniformStrat)
       
       #% Compute the background gradients in physical 2D space
       dUdz = np.expand_dims(duz, axis=1)
       DUDZ = np.tile(dUdz, NX+1)
       DUDZ = computeColumnInterp(DIM0, REF0[1], dUdz, 0, ZTL, DUDZ, CHT, interpolationType)
       # Compute thermodynamic gradients (no interpolation!)
       PORZ = PHYS[3] * TZ
       DLPDZ = -PHYS[0] / PHYS[3] * np.reciprocal(TZ)
       DLTDZ = np.reciprocal(TZ) * DTDZ
       DLPTDZ = DLTDZ - PHYS[4] * DLPDZ
       # Compute 2nd derivatives
       D2LPDZ2 = - DLTDZ * DLPDZ
       D2LPTDZ2 = np.reciprocal(TZ) * D2TDZ2 - DLTDZ * DLPTDZ
       
       # Get the static vertical gradients and store
       DUDZ = np.reshape(DUDZ, (OPS,1), order='F')
       DLTDZ = np.reshape(DLTDZ, (OPS,1), order='F')
       DLPDZ = np.reshape(DLPDZ, (OPS,1), order='F')
       DLPTDZ = np.reshape(DLPTDZ, (OPS,1), order='F')
       DQDZ = np.hstack((DUDZ, np.zeros((OPS,1)), DLPDZ, DLPTDZ))
       
       # Compute the background (initial) fields
       U = np.expand_dims(uz, axis=1)
       UZ = np.tile(U, NX+1)
       UZ = computeColumnInterp(DIM0, REF0[1], U, 0, ZTL, UZ, CHT, interpolationType)
       LPZ = np.expand_dims(lpz, axis=1)
       LOGP = np.tile(LPZ, NX+1)
       LOGP = computeColumnInterp(DIM0, REF0[1], LPZ, 0, ZTL, LOGP, CHT, interpolationType)
       LPT = np.expand_dims(lpt, axis=1)
       LOGT = np.tile(LPT, NX+1)
       LOGT = computeColumnInterp(DIM0, REF0[1], LPT, 0, ZTL, LOGT, CHT, interpolationType)       
       PBAR = np.exp(LOGP) # Hydrostatic pressure
       
       #%% RAYLEIGH AND GML WEIGHT OPERATORS
       ROPS, RLM, GML, SBR = computeRayleighEquations(DIMS, REFS, ZRL, RLOPT, ubdex, utdex)
       if ApplyGML:
              GMLOP = sps.diags(np.reshape(GML[0], (OPS,), order='F'), offsets=0, format='csr')
              GMLOX = sps.diags(np.reshape(GML[1], (OPS,), order='F'), offsets=0, format='csr')
              GMLOZ = sps.diags(np.reshape(GML[2], (OPS,), order='F'), offsets=0, format='csr')
       else:
              GMLOP = sps.identity(OPS, format='csr')
              GMLOX = sps.identity(OPS, format='csr')
              GMLOZ = sps.identity(OPS, format='csr')
       
       SBROP = sps.diags(np.reshape(SBR, (OPS,), order='F'), offsets=0, format='csr')
       # Make a collection for background field derivatives
       REFG = [(GMLOP, GMLOX, GMLOZ), DLTDZ, DQDZ, RLOPT[4], RLM, SBROP]
       
       # Update the REFS collection
       REFS.append(np.reshape(UZ, (OPS,), order='F'))
       REFS.append((np.reshape(PORZ, (OPS,), order='F'), np.reshape(PBAR, (OPS,), order='F')))
       
       # Get some memory back here
       del(PORZ)
       del(DUDZ)
       del(DLTDZ)
       del(DLPDZ)
       del(DLPTDZ)
       del(GML)
       del(REF0)
       del(DIM0)
       
       #%% DIFFERENTIATION OPERATORS
       
       DDX_QS, DDX4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[0], DDX_1D)
       DDZ_QS, DDZ4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[1], DDZ_1D)
       
       DDXMS, DDZMS = devop.computePartialDerivativesXZ(DIMS, REFS, DDX_1D, DDZ_1D)
       
       # Cubic Spline first derivative matrix
       DDXM_CS, DDZM_CS = devop.computePartialDerivativesXZ(DIMS, REFS, DDX_QS, DDZ_QS)
       
       # Make the TF operators
       DZDX = np.reshape(DZT, (OPS,1), order='F')
       
       # Prepare derivative operators for diffusion
       from rsb import rsb_matrix
       diffOps1 = (DDXM_CS, DDZM_CS)
       diffOps2 = (rsb_matrix(DDXM_CS, shape=DDXM_CS.shape), 
                   rsb_matrix(DDZM_CS, shape=DDZM_CS.shape))
       
       REFS.append((DDXMS, DDZMS)) # index 10
       REFS.append(diffOps1) # index 11
       
       if not StaticSolve and RSBops:
              # Multithreaded enabled for transient solution
              REFS.append((rsb_matrix(DDXMS,shape=DDXMS.shape), \
                           rsb_matrix(DDZMS,shape=DDZMS.shape)))
              REFS.append(diffOps2)
       elif not StaticSolve and not RSBops:
              # Native sparse
              REFS.append((DDXMS, DDZMS))
              REFS.append(diffOps1)
       else: 
              # Matrix operators for staggered method...
              REFS.append(DDXMS)
              REFS.append(DDZMS)
              
       # Store the terrain profile
       REFS.append(DZT)
       REFS.append(DZDX)
       REFS.append(DDXM_CS.dot(DZDX))
       REFS.append(np.power(DZDX, 2.0))
       
       # Compute and store the 2nd derivatives of background quantities
       D2QDZ2 = DDZM_CS.dot(DQDZ)
       D2QDZ2[:,2] = np.reshape(D2LPDZ2, (OPS,), order='F')
       D2QDZ2[:,3] = np.reshape(D2LPTDZ2, (OPS,), order='F')
       REFG.append(D2QDZ2)
       
       if not StaticSolve:
              NL = 6 # Number of eigenvalues to inspect...
              #'''
              print('Computing spectral radii of derivative operators...')
              PPXM = (DDXMS - sps.diags(DZDX[:,0]).dot(DDZMS)).tolil()
              DXE = PPXM[np.ix_(ebcDex[2],ebcDex[2])].tocsr()
              DZE = HOPT[0] / DIMS[2] * DDZMS[np.ix_(ebcDex[0],ebcDex[0])].tocsr()
              
              DX_eig = spl.eigs(DXE[1:-1,1:-1], k=NL, which='LM', return_eigenvectors=False)
              DZ_eig = spl.eigs(DZE[0:-1,0:-1], k=NL, which='LM', return_eigenvectors=False)
              
              print('Eigenvalues (largest magnitude) of derivative matrices:')
              print('X: ', DX_eig)
              print('Z: ', DZ_eig)
              
              DXI = np.imag(DX_eig)
              DZI = np.imag(DZ_eig)
              print('Eigenvalues size (largest magnitude) of derivative matrices:')
              print('X: ', np.abs(DXI))
              print('Z: ', np.abs(DZI))
              
              # Minimum magnitude eigenvalues to "cover" smallest resolved scale 
              DX_rho = np.amax(np.abs(DXI))
              DZ_rho = np.amax(np.abs(DZI))
              
              print('Derivative matrix spectral radii (1/m):')
              print('X: ', DX_rho)
              print('Z: ', DZ_rho)
              
              DX_spr = 1.0 / DX_rho
              DZ_spr = 1.0 / DZ_rho
              
              print('Grid resolution based on 1st derivative matrices: ')
              print('X: ', DX_spr)
              print('Z: ', DZ_spr)
              
              # Diffusion filter grid length based on resolution powers
              if isSEM_X and isSEM_Z:
                     DLD = (2.0 * DX_spr, 2.0 * DZ_spr)
                     DX = DX_min; DZ = DZ_min
              elif isSEM_X and not isSEM_Z:
                     DLD = (1.0 * DX_max, 1.0 * DZ_spr)
                     DX = DX_min; DZ = DZ_min
              elif not isSEM_X and isSEM_Z:
                     DLD = (1.0 * DX_spr, 1.0 * DZ_max)
                     DX = DX_min; DZ = DZ_min
              else:
                     DLD = (2.0 * DX_spr, 2.0 * DZ_spr)
                     DX = DX_min; DZ = DZ_min
                     
              DLD2 = DLD[0] * DLD[1]
              
              print('Diffusion lengths: ', DLD[0], DLD[1])
              
              del(PPXM); #del(PPZM)
              del(DXE); del(DZE)
              
              # Smallest physical grid spacing in the 2D mesh
              DLS = min(DX, DZ)
              #'''              
       del(DDXMS); del(DDXM_CS)
       del(DDZMS); del(DDZM_CS)
       del(DZDX); del(GMLOP)
       #input('STOP')
       
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
       fields, U, W = \
              eqs.computePrepareFields(REFS, currentState, INIT, udex, wdex, pdex, tdex)
              
       # NetCDF restart
       isRestartFromNC = True
       if isRestartFromNC and NonLinSolve:
              try:
                     rdex = -1
                     fname = 'transientNL_Spectral_QS.nc'
                     m_fid = Dataset(fname, 'r', format="NETCDF4")
                     thisTime = m_fid.variables['t'][rdex]
                     fields[:,0] = np.reshape(m_fid.variables['u'][rdex,:,:], (OPS,), order='F')
                     fields[:,1] = np.reshape(m_fid.variables['w'][rdex,:,:], (OPS,), order='F')
                     fields[:,2] = np.reshape(m_fid.variables['ln_p'][rdex,:,:], (OPS,), order='F')
                     fields[:,3] = np.reshape(m_fid.variables['ln_t'][rdex,:,:], (OPS,), order='F')
                     
                     DCF[0][:,0] = np.reshape(m_fid.variables['CRES_X'][rdex,:,:], (OPS,), order='F')
                     DCF[1][:,0] = np.reshape(m_fid.variables['CRES_Z'][rdex,:,:], (OPS,), order='F')
              except:
                     print('Could NOT read restart NC file!', fname)
       else:
             thisTime = IT 
              
       # Initialize output to NetCDF
       hydroState = np.reshape(INIT, (OPS, numVar), order='F')
       m_fid, tmvar, uvar, wvar, pvar, tvar, duvar, dwvar, dpvar, dtvar, dvar0, dvar1 = \
              initializeNetCDF(thisTime, NX, NZ, XL, ZTL, hydroState)
              
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
              DqDx, DqDz = \
                     eqs.computeFieldDerivatives(fields, REFS[10][0], REFS[10][1])
              rhs = eqs.computeEulerEquationsLogPLogT_StaticResidual(PHYS, DqDx, DqDz, REFG, \
                                                         REFS[15], REFS[9][0], fields, U, W, ebcDex, zeroDex)
              rhs += eqs.computeRayleighTendency(REFG, fields, zeroDex)
              
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
                     factorDS = dsl.lu_factor(DS, overwrite_a=True, check_finite=False)
                     del(DS)
                     print('Factor D... DONE!')
                     
                     # Store factor_DS for a little bit...
                     FDS = shelve.open(localDir + 'factorDS', flag='n', protocol=4)
                     FDS['factorDS'] = factorDS
                     FDS.close()
                     print('Store LU factor of D... DONE!')
                     
                     # Compute f2_hat = DS^-1 * f2 and f1_hat
                     BS = computeSchurBlock(schurName,'BS')
                     f2_hat = dsl.lu_solve(factorDS, f2)
                     f1_hat = f1 - BS.dot(f2_hat)
                     del(f1)
                     del(BS) 
                     del(f2_hat)
                     print('Compute modified force vectors... DONE!')
                     
                     # Get CS block and store in column chunks
                     CS = computeSchurBlock(schurName, 'CS')
                     fileCS = localDir + 'CS'
                     NCPU, CS_cranges = storeColumnChunks(CS, 'CS', fileCS)
                     print('Partition block C into chunks and store... DONE!')
                     del(CS)
                     
                     # Get AS block and store in column chunks
                     AS = computeSchurBlock(schurName, 'AS')
                     fileAS = localDir + 'AS'
                     NCPU, AS_cranges = storeColumnChunks(AS, 'AS', fileAS)
                     print('Partition block A into chunks and store... DONE!')
                     del(AS)
                     
                     # Loop over the chunks from disk
                     #AS = computeSchurBlock(schurName, 'AS')
                     BS = computeSchurBlock(schurName, 'BS')
                     ASmdb = shelve.open(fileAS)
                     CSmdb = shelve.open(fileCS, flag='r')
                     print('Computing DS^-1 * CS in chunks: ', NCPU)
                     for cc in range(NCPU):
                            # Get CS chunk
                            #CS_crange = CS_cranges[cc] 
                            CS_chunk = CSmdb['CS' + str(cc)]
                            
                            DS_chunk = dsl.lu_solve(factorDS, CS_chunk, overwrite_b=True, check_finite=False) # LONG EXECUTION
                            del(CS_chunk)
                            
                            # Get AS chunk
                            #AS_crange = AS_cranges[cc] 
                            AS_chunk = ASmdb['AS' + str(cc)]
                            #AS[:,crange] -= BS.dot(DS_chunk) # LONG EXECUTION
                            ASmdb['AS' + str(cc)] = AS_chunk - BS.dot(DS_chunk)
                            del(AS_chunk)
                            del(DS_chunk)
                            print('Computed chunk: ', cc+1)
                            
                     CSmdb.close()
                     del(BS)
                     del(factorDS)
                     
                     # Reassemble Schur complement of DS from AS chunk storage
                     print('Computing Schur Complement of D from chunks.')
                     DS_SC = ASmdb['AS0']
                     for cc in range(1,NCPU):
                            DS_SC = np.hstack((DS_SC, ASmdb['AS' + str(cc)]))
                     ASmdb.close()
                     print('Solve DS^-1 * CS... DONE!')
                     print('Compute Schur Complement of D... DONE!')
                     #'''
                     # Apply Schur C. solver on block partitioned DS_SC
                     factorDS_SC = dsl.lu_factor(DS_SC, overwrite_a=True)
                     del(DS_SC)
                     print('Factor Schur Complement of D... DONE!')
                     #'''
                     sol1 = dsl.lu_solve(factorDS_SC, f1_hat, overwrite_b=True, check_finite=False)
                     del(factorDS_SC)
                     #sol1, icode = spl.bicgstab(AS, f1_hat)
                     del(f1_hat)
                     print('Solve for u and w... DONE!')
                     
                     CS = computeSchurBlock(schurName, 'CS')
                     f2_hat = f2 - CS.dot(sol1)
                     del(f2)
                     del(CS)
                     FDS = shelve.open(localDir + 'factorDS', flag='r', protocol=4)
                     factorDS = FDS['factorDS']
                     FDS.close()
                     sol2 = dsl.lu_solve(factorDS, f2_hat, overwrite_b=True, check_finite=False)
                     del(f2_hat)
                     del(factorDS)
                     print('Solve for ln(p) and ln(theta)... DONE!')
                     dsol = np.concatenate((sol1, sol2))
                     
                     # Get memory back
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
              fields, U, W = \
                     eqs.computePrepareFields(REFS, np.array(SOLT[:,0]), INIT, udex, wdex, pdex, tdex)
              
              #%% Set the output residual and check
              message = 'Residual 2-norm BEFORE Newton step:'
              err = displayResiduals(message, RHS, 0.0, udex, wdex, pdex, tdex)
              DqDx, DqDz = \
                     eqs.computeFieldDerivatives(fields, REFS[10][0], REFS[10][1])
              rhs = eqs.computeEulerEquationsLogPLogT_StaticResidual(PHYS, DqDx, DqDz, REFG, \
                                                         REFS[15], REFS[9][0], fields, U, W, ebcDex, zeroDex)
              rhs += eqs.computeRayleighTendency(REFG, fields, zeroDex)
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
              
              if TOPT[3] == 3:
                     DTF = 1.25
              elif TOPT[3] == 4:
                     DTF = 1.5
              else:
                     DTF = 1.1
              
              # Initialize the perturbations
              if thisTime == 0.0:
                     # Initialize damping coefficients
                     DCF = (np.zeros((OPS,1)), np.zeros((OPS,1)))
                     
                     # Initialize vertical velocity
                     fields[ubdex,1] = -dWBC
                     
                     PTR = np.power(np.power((XL + 100.0E3) / 30.0E3, 2.0) + \
                            np.power((ZTL - 10.0E3) / 5.0E3, 2.0), 0.5)
                     PTF = np.power(np.cos(0.5 * mt.pi * PTR), 2.0)
                     PTF = 10.0 * np.where(PTR <= 1.0, PTF, 0.0)
                     
                     LPTF = np.log(1.0 + PTF * np.reciprocal(np.exp(LOGT)) )
                     
                     fields[:,3] = np.reshape(LPTF, (OPS,), order='F')
              
              # Initialize local sound speed and time step
              #'''
              UD = fields[:,0] + hydroState[:,0]
              WD = fields[:,1]
              VSND = np.sqrt(PHYS[6] * REFS[9][0])
              VWAV_max = bn.nanmax(VSND)
              DT0 = DTF * DLS / VWAV_max
              TOPT[0] = DT0
              print('Initial time step by sound speed: ', str(DT0) + ' (sec)')
              print('Time stepper order: ', str(TOPT[3]))
              print('Time step factor: ', str(DTF))
              
              OTI = int(TOPT[5] / DT0)
              ITI = int(TOPT[6] / DT0)
              #'''
              
              ti = 0; ff = 0
              delFields = np.zeros(fields.shape)
              rhsVec = np.zeros(fields.shape)
              resVec = np.zeros(fields.shape)
              error = [np.linalg.norm(rhsVec)]
              rhsVec0 = np.copy(rhsVec)
              delFields0 = np.copy(delFields)
              
              while thisTime <= TOPT[4]:
                     
                     if ti == 0:
                            isFirstStep = True
                     else:
                            isFirstStep = False
                             
                     # Print out diagnostics every TOPT[5] steps
                     if ti % OTI == 0:
                            message = ''
                            err = displayResiduals(message, np.reshape(rhsVec, (OPS*numVar,), order='F'), thisTime, udex, wdex, pdex, tdex)
                            error.append(err)
                            
                            if makePlots:
                                   makeFieldPlots(TOPT, thisTime, XL, ZTL, fields, rhsVec, resVec, NX, NZ, numVar)
                     
                     if ti % ITI == 0:
                            # Store current time to NC file
                            tmvar[ff] = thisTime
                            # Check the fields or tendencies
                            for pp in range(numVar):
                                   q = np.reshape(fields[:,pp], (NZ+1, NX+1), order='F')
                                   dqdt = np.reshape(rhsVec[:,pp], (NZ+1, NX+1), order='F')

                                   if pp == 0:
                                          uvar[ff,:,:] = q
                                          duvar[ff,:,:] = dqdt
                                   elif pp == 1:
                                          wvar[ff,:,:] = q
                                          dwvar[ff,:,:] = dqdt
                                   elif pp == 2:
                                          pvar[ff,:,:] = q
                                          dpvar[ff,:,:] = dqdt
                                   else:
                                          tvar[ff,:,:] = q
                                          dtvar[ff,:,:] = dqdt
                                          
                            dvar0[ff,:,:] = np.reshape(DCF[0], (NZ+1, NX+1), order='F')
                            dvar1[ff,:,:] = np.reshape(DCF[1], (NZ+1, NX+1), order='F')
                            ff += 1
                            
                     import computeResidualViscCoeffs as rescf
                     # Compute the solution within a time step
                     try:   
                            # Compute a time step
                            delFields = tint.computeTimeIntegrationNL2(DIMS, PHYS, REFS, REFG, \
                                                                    DLD, DLD2,\
                                                                    TOPT, fields, delFields, hydroState, \
                                                                    zeroDex, ebcDex, \
                                                                    DynSGS, DCF, isFirstStep)
                            
                            # Apply update
                            fields += delFields
                            thisTime += TOPT[0]
                            Q = fields + hydroState
                                   
                            # Compute flow speed
                            UD = Q[:,0]
                            WD = Q[:,1]
                            #'''
                            vel = np.stack((UD, WD),axis=1)
                            VFLW = np.linalg.norm(vel, axis=1)
                            
                            # Compute the updated RHS
                            DqDx, DqDz = \
                                   eqs.computeFieldDerivatives(fields, REFS[12][0], REFS[12][1])
                            rhsVec = eqs.computeEulerEquationsLogPLogT_Explicit(PHYS, DqDx, DqDz, REFG, REFS[15], REFS[9][0], \
                                                                          fields, UD, WD, ebcDex, zeroDex)
                            rhsVec += eqs.computeRayleighTendency(REFG, fields, zeroDex)
                            
                            # Compute the current residual
                            if ti == 0:
                                   resVec = (1.0 / TOPT[0]) * delFields - rhsVec
                            else:
                                   resVec = (0.5 / DT0) * delFields0 + \
                                          (0.5 / TOPT[0]) * delFields - \
                                          0.5 * (rhsVec0 + rhsVec)
                                          #(0.5 * rhsVec0 - 1.5 * rhsVec)
                            
                            # Compute DynSGS or Flow Dependent diffusion coefficients
                            #QM = bn.nanmax(np.abs(fields), axis=0)
                            DQ = fields - np.mean(fields, axis=0)
                            QM = bn.nanmax(np.abs(DQ), axis=0)
                            #newDiff = rescf.computeResidualViscCoeffs(DIMS, resVec, QM, VFLW, DLD, DLD2, 'maximum', NE)
                            newDiff = rescf.computeResidualViscCoeffs3(DIMS, resVec, QM, VFLW, DLD, DLD2, NE)
                            
                            DCF[0][:,0] = newDiff[0]
                            DCF[1][:,0] = newDiff[1]
                            
                            # Get the previous state
                            DT0 = TOPT[0]
                            rhsVec0 = np.copy(rhsVec)
                            delFields0 = np.copy(delFields)
                            
                            # Compute sound speed
                            T_ratio = np.expm1(PHYS[4] * fields[:,2] + fields[:,3])
                            RdT = REFS[9][0] * (1.0 + T_ratio)
                            VSND = np.sqrt(PHYS[6] * RdT)
                            
                            # Compute new time step based on updated sound speed
                            DTN = DTF * DLS / bn.nanmax(VSND)
                            
                            # Update the local time step
                            TOPT[0] = DTN
                            
                     except Exception:
                            print('Transient step failed! Closing out to NC file. Time: ', thisTime)
                            
                            m_fid.close() 
                            makeFieldPlots(TOPT, thisTime, XL, ZTL, fields, rhsVec, resVec, NX, NZ, numVar)
                            import traceback
                            traceback.print_exc()
                            sys.exit(2)
                     #'''
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
              rdb['qdex'] = (udex, wdex, pdex, tdex)
              rdb['INIT'] = INIT
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
       uxz = np.reshape(SOLT[udex,0], (NZ+1,NX+1), order='F') 
       wxz = np.reshape(SOLT[wdex,0], (NZ+1,NX+1), order='F')
       pxz = np.reshape(SOLT[pdex,0], (NZ+1,NX+1), order='F') 
       txz = np.reshape(SOLT[tdex,0], (NZ+1,NX+1), order='F')
       
       #%% Make some plots for static or transient solutions
       if makePlots and StaticSolve:
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
                     dqdt = np.reshape(RHS[qdex], (NZ+1, NX+1), order='F')
                     ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 201, cmap=cm.seismic)
                     plt.colorbar(ccheck, format='%+.3E')
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                     plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
                     plt.tight_layout()
              plt.show()
              
              # Check W at the boundaries...
              dwdt = np.reshape(RHS[wdex], (NZ+1, NX+1), order='F')
              return (XL, ZTL, dwdt)
       
if __name__ == '__main__':
       
       #TestName = 'ClassicalSchar01'
       #TestName = 'ClassicalScharIter'
       #TestName = 'SmoothStratScharIter'
       #TestName = 'DiscreteStratScharIter'
       #TestName = '3LayerTest'
       TestName = 'UniformTest'
       
       # Run the model in a loop if needed...
       for ii in range(1):
              diagOutput = runModel(TestName)