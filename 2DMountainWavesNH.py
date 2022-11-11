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
# Set up the multithreading environment (physical cores only)
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import sys
import time
import shelve
import shutil
import math as mt
import numpy as np
import bottleneck as bn
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import scipy.spatial as spk
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
import computeResidualViscCoeffs as rescf

# Numerical stuff
import computeDiskPartSchur as dsolver
import computeDerivativeMatrix as derv
import computeEulerEquationsLogPLogT as eqs
import computeTimeIntegration as tint

import faulthandler; faulthandler.enable()

from netCDF4 import Dataset

# Disk settings
#localDir = '/scratch/opbuffer/' # NOAA laptop
localDir = '/home/scratch/' # Home super desktop
#localDir = '/Users/TempestGuerra/scratch/' # Davis Macbook Pro
#localDir = '/home/jeguerra/scratch/'
restart_file = localDir + 'restartDB'
schurName = localDir + 'SchurOps'
fname2Restart = 'Simulation2Restart.nc'
fname4Restart = 'SimulationTemp.nc'

def makeTemperatureBackgroundPlots(Z_in, T_in, ZTL, TZ, DTDZ):
       
       # Make a figure of the temperature background
       plt.figure(figsize=(18.0, 6.0))
       plt.subplot(1,3,1)
       plt.plot(T_in, 1.0E-3*np.array(Z_in), 'ko-')
       plt.title('Discrete Temperature Profile (K)')
       plt.xlabel('Temperature (K)')
       plt.ylabel('Height (km)')
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,3,2)
       plt.plot(TZ, 1.0E-3*ZTL, 'k-')
       plt.title('Smooth Temperature Profile (K)')
       plt.xlabel('Temperature (K)')
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,3,3)
       plt.plot(DTDZ, 1.0E-3*ZTL, 'k-')
       plt.title('Smooth Temperature Derivative (K/z)')
       plt.xlabel('Temperature (K)')
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       
       plt.tight_layout()
       plt.savefig('python results/Temperature_Background.png')
       plt.show()
       sys.exit(2)

       return       

def makeFieldPlots(TOPT, thisTime, XL, ZTL, fields, rhs, res, dca, dcb, NX, NZ, numVar):
       
       keepHistory = False
       
       xlims = [-50.0, 50.0]
       ylims = [0.0, 25.0]
       
       fig = plt.figure(num=1, clear=True, figsize=(13.0, 9.0)) 
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
              plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
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
                     
              plt.xlim(xlims[0], xlims[1])
              plt.ylim(ylims[0], ylims[1])
                     
              #%% Plot the full tendencies
              plt.subplot(4,3,rowDex + 1)
              if np.abs(dqdt.max()) > np.abs(dqdt.min()):
                     clim = np.abs(dqdt.max())
              elif np.abs(dqdt.max()) < np.abs(dqdt.min()):
                     clim = np.abs(dqdt.min())
              else:
                     clim = np.abs(dqdt.max())
             
              ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 101, cmap=cm.seismic, vmin=-clim, vmax=clim)
              plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
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
                     
              plt.xlim(xlims[0], xlims[1])
              plt.ylim(ylims[0], ylims[1])
                     
              #%% Plot the full residuals
              plt.subplot(4,3,rowDex + 2)
              if np.abs(residual.max()) > np.abs(residual.min()):
                     clim = np.abs(residual.max())
              elif np.abs(residual.max()) < np.abs(residual.min()):
                     clim = np.abs(residual.min())
              else:
                     clim = np.abs(residual.max())
                 
              normr = 1.0
              ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, normr * residual, 101, cmap=cm.seismic, vmin=-clim, vmax=clim)
              plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
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
                     
              plt.xlim(xlims[0], xlims[1])
              plt.ylim(ylims[0], ylims[1])
                     
       plt.tight_layout()
       if keepHistory:
              plt.savefig('/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/solutions' + '{:0>6d}'.format(int(thisTime)) +  '.pdf')
       else:
              shutil.copyfile('/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/solutions.pdf', \
                              '/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/solutions0.pdf')
              plt.savefig('/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/solutions.pdf')
       fig.clear()
       del(fig)
       
       #%% PLOT DIFFUCION COEFFICIENTS
       fig = plt.figure(num=1, clear=True, figsize=(13.0, 9.0)) 
       d1_rhs = np.reshape(dca[0], (NZ+1, NX+1), order='F')
       d2_rhs = np.reshape(dca[1], (NZ+1, NX+1), order='F')
       d1_res = np.reshape(dcb[0], (NZ+1, NX+1), order='F')
       d2_res = np.reshape(dcb[1], (NZ+1, NX+1), order='F')
       
       plt.subplot(2,2,1)
       cc1 = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, d1_rhs, 51, cmap=cm.afmhot)
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.colorbar(cc1, format='%.2e')
       plt.title('RHS Coefficients X')
       plt.subplot(2,2,2)
       cc2 = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, d2_rhs, 51, cmap=cm.afmhot)
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.colorbar(cc2, format='%.2e')
       plt.title('RHS Coefficients Z')
       plt.subplot(2,2,3)
       cc3 = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, d1_res, 51, cmap=cm.afmhot,)
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.colorbar(cc3, format='%.2e')
       plt.title('RES Coefficients X')
       plt.subplot(2,2,4)
       cc4 = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, d2_res, 51, cmap=cm.afmhot)
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.colorbar(cc4, format='%.2e')
       plt.title('RES Coefficients Z')
       
       plt.tight_layout()
       if keepHistory:
              plt.savefig('/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/diffusions' + '{:0>6d}'.format(int(thisTime)) +  '.pdf')
       else:
              shutil.copyfile('/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/diffusions.pdf', \
                              '/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/diffusions0.pdf')
              plt.savefig('/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/diffusions.pdf')
       fig.clear()
       del(fig)
       
       return

def displayResiduals(message, RHS, thisTime, DT, OPS, udex, wdex, pdex, tdex):
       err = np.linalg.norm(RHS) / mt.sqrt(OPS)
       err1 = np.linalg.norm(RHS[udex]) / mt.sqrt(OPS)
       err2 = np.linalg.norm(RHS[wdex]) / mt.sqrt(OPS)
       err3 = np.linalg.norm(RHS[pdex]) / mt.sqrt(OPS)
       err4 = np.linalg.norm(RHS[tdex]) / mt.sqrt(OPS)
       if message != '':
              print(message)
       print('Time (min): %4.2f, DT (sec): %4.3E, Residuals: %10.4E, %10.4E, %10.4E, %10.4E, %10.4E' \
             % (thisTime / 60.0, DT, err1, err2, err3, err4, err))
       
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

def initializeNetCDF(fname, thisTime, NX, NZ, XL, ZTL, hydroState):
       # Rename output file to the current time for subsequent storage
       if thisTime > 0.0:
              newFname = fname[0:-3] + str(int(thisTime)) + '.nc'
       else:
              newFname = fname
       
       try:
              m_fid = Dataset(newFname, 'w', format="NETCDF4")
              print('Data output to: ', newFname)
       except PermissionError:
              print('Deleting corrupt NC file... from failed run.')
              import os
              os.remove(fname)
              m_fid = Dataset(newFname, 'w', format="NETCDF4")
              
       # Make dimensions
       m_fid.createDimension('time', None)
       m_fid.createDimension('x', NX+1)
       m_fid.createDimension('y', 1)
       m_fid.createDimension('z', NZ+1)
       # Create variables (time and grid)
       tmvar = m_fid.createVariable('time', 'f8', ('time',))
       tmvar.units = 'seconds'
       tmvar.axis = 'T'
       xgvar = m_fid.createVariable('Xlon', 'f8', ('z', 'x', 'y'))
       xgvar.units = 'm'
       ygvar = m_fid.createVariable('Ylat', 'f8', ('z', 'x', 'y'))
       ygvar.units = 'm'
       zgvar = m_fid.createVariable('Zhgt', 'f8', ('z', 'x', 'y'))
       zgvar.units = 'm'
       # Store variables
       xgvar[:,:,0] = XL
       ygvar[:,:,0] = 0.0 * XL
       zgvar[:,:,0] = ZTL
       # Create variables (background static fields)
       UVAR = m_fid.createVariable('U', 'f8', ('z', 'x', 'y'))
       PVAR = m_fid.createVariable('LNP', 'f8', ('z', 'x', 'y'))
       TVAR = m_fid.createVariable('LNT', 'f8', ('z', 'x', 'y'))
       # Store variables
       UVAR[:,:,0] = np.reshape(hydroState[:,0], (NZ+1,NX+1), order='F')
       PVAR[:,:,0] = np.reshape(hydroState[:,2], (NZ+1,NX+1), order='F')
       TVAR[:,:,0] = np.reshape(hydroState[:,3], (NZ+1,NX+1), order='F')
       # Create variables (transient fields)
       m_fid.createVariable('u', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('w', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('ln_p', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('ln_t', 'f8', ('time', 'z', 'x', 'y'))
       # Create variables (field tendencies)
       m_fid.createVariable('DuDt', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('DwDt', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Dln_pDt', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Dln_tDt', 'f8', ('time', 'z', 'x', 'y'))
       # Create variables (field residuals)
       m_fid.createVariable('Ru', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Rw', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Rln_p', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Rln_t', 'f8', ('time', 'z', 'x', 'y'))
       # Create variables for diffusion coefficients
       m_fid.createVariable('DC1', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('DC2', 'f8', ('time', 'z', 'x', 'y'))
       #m_fid.createVariable('DCln_p', 'f8', ('time', 'z', 'x', 'y'))
       #m_fid.createVariable('DCln_t', 'f8', ('time', 'z', 'x', 'y'))
       
       m_fid.close()
       del(m_fid)
       
       return newFname

def store2NC(newFname, thisTime, ff, numVar, NX, NZ, fields, rhsVec, resVec, DCF):
       # Store in the NC file
       try:
              m_fid = Dataset(newFname, 'a', format="NETCDF4")
              m_fid.variables['time'][ff] = thisTime
       
              for pp in range(numVar):
                     q = np.reshape(fields[:,pp], (NZ+1, NX+1), order='F')
                     dqdt = np.reshape(rhsVec[:,pp], (NZ+1, NX+1), order='F')
                     rq = np.reshape(resVec[:,pp], (NZ+1, NX+1), order='F')
                     #dq = np.reshape(DCF[0][:,pp], (NZ+1, NX+1), order='F')
                     dq1 = np.reshape(DCF[0][:,0], (NZ+1, NX+1), order='F')
                     dq2 = np.reshape(DCF[1][:,0], (NZ+1, NX+1), order='F')

                     if pp == 0:
                            m_fid.variables['u'][ff,:,:,0] = q
                            m_fid.variables['DuDt'][ff,:,:,0] = dqdt
                            m_fid.variables['Ru'][ff,:,:,0] = rq
                            m_fid.variables['DC1'][ff,:,:,0] = dq1
                     elif pp == 1:
                            m_fid.variables['w'][ff,:,:,0] = q
                            m_fid.variables['DwDt'][ff,:,:,0] = dqdt
                            m_fid.variables['Rw'][ff,:,:,0] = rq
                            m_fid.variables['DC2'][ff,:,:,0] = dq2
                     elif pp == 2:
                            m_fid.variables['ln_p'][ff,:,:,0] = q
                            m_fid.variables['Dln_pDt'][ff,:,:,0] = dqdt
                            m_fid.variables['Rln_p'][ff,:,:,0] = rq
                            #m_fid.variables['DCln_p'][ff,:,:,0] = dq
                     else:
                            m_fid.variables['ln_t'][ff,:,:,0] = q
                            m_fid.variables['Dln_tDt'][ff,:,:,0] = dqdt
                            m_fid.variables['Rln_t'][ff,:,:,0] = rq
                            #m_fid.variables['DCln_t'][ff,:,:,0] = dq
                            
              m_fid.close()
              del(m_fid)
       except Exception as e:
              print(e)
              print('Could NOT store state to NC file!', newFname)
              print('At time (min): ', thisTime / 60)
              
       return

def runModel(TestName):
       import TestCase
       
       thisTest = TestCase.TestCase(TestName)
       
       # Set the solution type (MUTUALLY EXCLUSIVE)
       StaticSolve = thisTest.solType['StaticSolve']
       NonLinSolve = thisTest.solType['NLTranSolve']
       NewtonLin = thisTest.solType['NewtonLin']
       ExactBC = thisTest.solType['ExactBC']
       
       # Set the use of a persistent hydrostatic background:
       withHydroState = True
       
       # Switch to use the PyRSB multithreading module (CPU multithreaded SpMV)
       if StaticSolve and not NonLinSolve:
              RSBops = False
              useGuellrich = True
              useUniformSt = False
       else:
              RSBops = False # Turn off PyRSB SpMV
              
       if RSBops:
              from rsb import rsb_matrix
       
       # Set the grid type
       HermFunc = thisTest.solType['HermFuncGrid']
       # Use the uniform grid fourier solution if not Hermite Functions
       if HermFunc:
              FourierLin = False
              print('Hermite Function grid in the horizontal.')
       else:
              FourierLin = False
              print('Uniform Fourier grid in the horizontal.')
              
       inlineDCF = True
       if inlineDCF:
              print('DynSGS coefficients updated inline with RK stage updates.')
       else:
              print('DynSGS coefficients updated on full time step updates.')
              
       # Set residual diffusion switch
       DynSGS = thisTest.solType['DynSGS']
       if DynSGS:
              print('DynSGS Diffusion Model.')
       else:
              print('Flow-Dependent Diffusion Model.')
              
       # Spatial filtering for DynSGS coefficients
       filteredCoeffs = True
       if filteredCoeffs:
              print('Spatially filtered DynSGS coefficients.')
       else:
              print('No spatial filter on DynSGS coefficients.')
              
       DynSGS_RES = True
       if DynSGS_RES:
              print('Diffusion coefficients by residual estimate.')
       else:
              print('Diffusion coefficients by RHS evaluation.')
              
       diffusiveFlux = True
       if diffusiveFlux:
              print('Diffusion by gradient of diffusive flux.')
       else:
              print('Diffusion by product of coefficients with 2nd derivatives.')
              
       verticalChebGrid = False
       verticalLegdGrid = True
       if verticalChebGrid and not verticalLegdGrid:
              print('Chebyshev spectral derivative in the vertical.')
       else:
              print('Legendre spectral derivative in the vertical.')
       
       verticalStagger = True
       if verticalStagger:
              useGuellrich = True
              useUniformSt = False
              print('Staggered spectral method in the vertical.')
       else:
              useGuellrich = True
              useUniformSt = False
              print('Colocated spectral method in the vertical.')
       
       # Set direct solution method (MUTUALLY EXCLUSIVE)
       SolveFull = thisTest.solType['SolveFull']
       SolveSchur = thisTest.solType['SolveSchur']
       
       # Set Newton solve initial and restarting parameters
       toRestart = thisTest.solType['ToRestart'] # Saves resulting state to restart database
       isRestart = thisTest.solType['IsRestart'] # Initializes from a restart database
       makePlots = thisTest.solType['MakePlots'] # Switch for diagnostic plotting
       
       if isRestart:
              rdex = -10
       
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
       
       # Time step scaling depending on RK solver
       DTF = TOPT[2]
        
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
              
       #%% COMPUTE STRATIFICATION AT HIGH RESOLUTION SPECTRAL
       DIM0 = [DIMS[0], DIMS[1], DIMS[2], DIMS[3], 2 * DIMS[4], DIMS[5]]
       REF0 = computeGrid(DIM0, HermFunc, FourierLin, verticalChebGrid, verticalLegdGrid)
       
       # Get the double resolution operator here
       if verticalChebGrid:
              DDZP, ITRANS = derv.computeChebyshevDerivativeMatrix(DIM0)
       elif verticalLegdGrid:
              DDZP, ITRANS = derv.computeLegendreDerivativeMatrix(DIM0)
                     
       DDXP, DDX2P = derv.computeCubicSplineDerivativeMatrix(REF0[0], True, False, 0.0)
       REF0.append(DDXP)
       REF0.append(DDZP)
       
       hx, dhx = computeTopographyOnGrid(REF0, HOPT)
       zRay = DIMS[2] - RLOPT[0]
       
       if useGuellrich:
              xl, ztl, dzt, sig, ZRL, DXM, DZM = \
                     coords.computeGuellrichDomain2D(DIM0, REF0[0], REF0[1], zRay, hx, dhx, StaticSolve)
       
       if useUniformSt:
              xl, ztl, dzt, sig, ZRL, DXM, DZM = \
                     coords.computeStretchedDomain2D(DIM0, REF0, zRay, hx, dhx)
       
       REF0.append(xl)
       REF0.append(ztl)
       REF0.append(dhx)
       REF0.append(sig)
       
       tz, dtz, dtz2 = \
              computeTemperatureProfileOnGrid(PHYS, REF0, Z_in, T_in, smooth3Layer, uniformStrat, RLOPT, StaticSolve)
              
       dlpz, lpz, pz, dlptz, lpt, pt, rho = \
              computeThermoMassFields(PHYS, DIM0, REF0, tz[:,0], dtz[:,0], 'sensible', uniformStrat, RLOPT, StaticSolve)
              
       uz, duz = computeShearProfileOnGrid(REF0, JETOPS, PHYS[1], pz, dlpz, uniformWind, linearShear)
       
       '''
       # Check background
       fig, ax = plt.subplots(nrows=2, ncols=2)
       ax[0,0].plot(REF0[1], pz)
       ax[0,1].plot(REF0[1], pt)
       ax[1,0].plot(REF0[1], uz)
       ax[1,1].plot(REF0[1], tz)
       plt.show()
       input('Check background profiles.')
       '''
       #%% SET UP THE GRID AND INDEX VECTORS
       REFS = computeGrid(DIMS, HermFunc, FourierLin, verticalChebGrid, verticalLegdGrid)
      
       #%% Compute the raw derivative matrix operators in alpha-xi computational space
       if HermFunc and not FourierLin:
              DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       elif FourierLin and not HermFunc:
              DDX_1D, dummy = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, 0.0)
              #DDX_1D, HF_TRANS = derv.computeFourierDerivativeMatrix(DIMS)
       else:
              DDX_BC = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 6)
              DDX_1D, DDX4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[0], True, False, DDX_BC)
                            
       if verticalChebGrid:
              DDZ_1D, VTRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
       
       if verticalLegdGrid:
              DDZ_1D, VTRANS = derv.computeLegendreDerivativeMatrix(DIMS)
              
       #%% Update the REFS collection
       REFS.append(DDX_1D) # index 2
       REFS.append(DDZ_1D) # index 3
       
       #% Read in topography profile or compute from analytical function
       HofX, dHdX = computeTopographyOnGrid(REFS, HOPT)
              
       # Make the 2D physical domains from reference grids and topography
       zRay = DIMS[2] - RLOPT[0]
       
       if useGuellrich:
              XL, ZTL, DZT, sigma, ZRL, DXM, DZM = \
                     coords.computeGuellrichDomain2D(DIMS, REFS[0], REFS[1], zRay, HofX, dHdX, StaticSolve)
       
       if useUniformSt:
              XL, ZTL, DZT, sigma, ZRL, DXM, DZM = coords.computeStretchedDomain2D(DIMS, REFS, zRay, HofX, dHdX)
       
       # Update the REFS collection
       REFS.append(XL) # index 4
       REFS.append(ZTL) # index 5
       REFS.append((dHdX, HofX)) # index 6
       REFS.append(sigma) # index 7
       
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
              computeTemperatureProfileOnGrid(PHYS, REFS, Z_in, T_in, smooth3Layer, uniformStrat, RLOPT, StaticSolve)
       
       #makeTemperatureBackgroundPlots(Z_in, T_in, ZTL, TZ, DTDZ)
       # Compute the initial fields by interolation
       '''
       Uz = spi.interp1d(REF0[1], uz, kind='cubic')
       dUdz = spi.interp1d(REF0[1], duz, kind='cubic')
       LPZ = spi.interp1d(REF0[1], lpz, kind='cubic')
       LPT = spi.interp1d(REF0[1], lpt, kind='cubic')
       
       UZ = np.zeros(ZTL.shape)
       DUDZ = np.zeros(ZTL.shape)
       LOGP = np.zeros(ZTL.shape)
       LOGT = np.zeros(ZTL.shape)
       for cc in range(ZTL.shape[1]):
              UZ[:,cc] = Uz(ZTL[:,cc])
              DUDZ[:,cc] = dUdz(ZTL[:,cc])
              LOGP[:,cc] = LPZ(ZTL[:,cc])
              LOGT[:,cc] = LPT(ZTL[:,cc])
       '''       
       dUdz = np.expand_dims(duz, axis=1)
       DUDZ = computeColumnInterp(DIM0, REF0[1], dUdz, 0, ZTL, ITRANS)
       
       # Compute thermodynamic gradients (no interpolation!)
       PORZ = PHYS[3] * TZ
       DLPDZ = -PHYS[0] / PHYS[3] * np.reciprocal(TZ)
       DLTDZ = np.reciprocal(TZ) * DTDZ
       DLPTDZ = DLTDZ - PHYS[4] * DLPDZ
       
       # Get the static vertical gradients and store
       DUDZ = np.reshape(DUDZ, (OPS,1), order='F')
       DLTDZ = np.reshape(DLTDZ, (OPS,1), order='F')
       DLPDZ = np.reshape(DLPDZ, (OPS,1), order='F')
       DLPTDZ = np.reshape(DLPTDZ, (OPS,1), order='F')
       DQDZ = np.hstack((DUDZ, np.zeros((OPS,1)), DLPDZ, DLPTDZ))
       
       # Compute the background (initial) fields
       U = np.expand_dims(uz, axis=1)
       UZ = computeColumnInterp(DIM0, REF0[1], U, 0, ZTL, ITRANS)
       LPZ = np.expand_dims(lpz, axis=1)
       LOGP = computeColumnInterp(DIM0, REF0[1], LPZ, 0, ZTL, ITRANS)
       LPT = np.expand_dims(lpt, axis=1)
       LOGT = computeColumnInterp(DIM0, REF0[1], LPT, 0, ZTL, ITRANS)       
       PBAR = np.exp(LOGP) # Hydrostatic pressure
       
       #plt.contourf(XL, ZTL, LOGT)
       #plt.show()
       #input()
       
       #%% RAYLEIGH AND GML WEIGHT OPERATORS
       ROPS, RLM, GML, LDEX = computeRayleighEquations(DIMS, REFS, ZRL, RLOPT, ebcDex)
       
       # Make a collection for background field derivatives
       REFG = [GML, DLTDZ, DQDZ, RLOPT[4], RLM, LDEX.flatten()]
              
       # Update the REFS collection
       REFS.append(np.reshape(UZ, (OPS,), order='F')) # index 8
       REFS.append((np.reshape(PORZ, (OPS,), order='F'), np.reshape(PBAR, (OPS,), order='F'))) #index 9
       
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
       DDX_CFD4 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 4)
       DDZ_CFD4 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 4)
       DDX_CS, DDX2_CS = derv.computeCubicSplineDerivativeMatrix(REFS[0], True, False, DDX_CFD4)
       DDZ_CS, DDZ2_CS = derv.computeCubicSplineDerivativeMatrix(REFS[1], True, False, DDZ_CFD4)
       
       DDX_CFD6 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 6)
       DDZ_CFD6 = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 6)
       DDX_QS, DDX4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[0], True, False, DDX_CFD6)
       DDZ_QS, DDZ4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[1], True, False, DDZ_CFD6)
       
       # Derivative operators for dynamics
       DDXMS1, DDZMS1 = devop.computePartialDerivativesXZ(DIMS, REFS[7], DDX_1D, DDZ_1D)
       DDXMS_CS, DDZMS_CS = devop.computePartialDerivativesXZ(DIMS, REFS[7], DDX_CS, DDZ_CS)
       DDXMS_QS, DDZMS_QS = devop.computePartialDerivativesXZ(DIMS, REFS[7], DDX_QS, DDZ_QS)
       DDXMS_CFD, DDZMS_CFD = devop.computePartialDerivativesXZ(DIMS, REFS[7], DDX_CFD6, DDZ_CFD6)
              
       #%% Staggered operator in the vertical Legendre/Chebyshev mix
       NZI = NZ + 1
       if verticalLegdGrid:
              
              xiI, whf = derv.cheblb(NZI) #[-1 1]
              xiO, whf = derv.leglb(NZ) #[-1 1]
              
              TMO = derv.chebpolym(NZI+1, xiO)
              TMI, dummy = derv.legpolym(NZ, xiI, True)
              
              # Get the outer vertical derivative
              DDZ_O0, O_TRANS = derv.computeLegendreDerivativeMatrix(DIMS)
              DDZ_O1, dummy = derv.computeQuinticSplineDerivativeMatrix(xiO, True, False, DDZ_O0)
              
              # Get the inner vertical derivatie
              DIMS_ST = [DIMS[0], DIMS[1], DIMS[2], DIMS[3], NZI, DIMS[5]]
              DDZ_I0, I_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS_ST)
              DDZ_I1, dummy = derv.computeQuinticSplineDerivativeMatrix(xiI, True, False, DDZ_I0)
              
              O2I_INT = (TMI.T).dot(VTRANS) # Outer to Internal grid
              I2O_INT = (TMO.T).dot(I_TRANS) # Inner to Outer grid
              
              DDZ_I = I2O_INT.dot(DDZ_I0).dot(O2I_INT)
              dummy, DDZMI = devop.computePartialDerivativesXZ(DIMS, REFS[7], DDX_1D, DDZ_I)
              dummy, DDZMO = devop.computePartialDerivativesXZ(DIMS, REFS[7], DDX_1D, DDZ_O0)
              
       if verticalStagger:
              # Staggered vertical operator
              DDZM_OP = sps.block_diag((DDZMO, DDZMI, DDZMO, DDZMI), format='csr')
       else:
              # Average the staggered operators
              #DDZM_OP = 1.0 * DDZMS1
              DDZM_OP = 0.5 * (DDZMO + DDZMI)
              
       if HermFunc:
              DDXM_OP = DDXMS1 - sps.diags(np.reshape(DZT, (OPS,), order='F')).dot(DDZMS1)
       else:
              DDXM_OP = DDXMS_QS - sps.diags(np.reshape(DZT, (OPS,), order='F')).dot(DDZMS_QS)
              
       #plt.plot(xi_lg, xi_lg, 'ko', xi_ch, xi_ch, 'ks')
       #plt.show()
       #input()
       
       #%% Prepare derivative operators for diffusion
       PPXMD = DDXMS_CS - sps.diags(np.reshape(DZT, (OPS,), order='F')).dot(DDZMS_CS)
       diffOps1 = (PPXMD, DDZMS_CS)
       
       # Get the operator for the terrain diffusion
       DDX_BC = np.copy(DDX_CS)
       
       REFS.append((DDXMS1, DDZMS1)) # index 10
       REFS.append(diffOps1) # index 11
       
       if not StaticSolve and RSBops:
              # Multithreaded enabled for transient solution
              REFS.append((rsb_matrix(DDXM_OP,shape=DDXM_OP.shape), \
                           rsb_matrix(DDZM_OP,shape=DDZM_OP.shape)))
              
              diffOps2 = (rsb_matrix(diffOps1[0], shape=diffOps1[0].shape), 
                          rsb_matrix(diffOps1[1], shape=diffOps1[1].shape))
              REFS.append(diffOps2)
              del(diffOps2)
       elif not StaticSolve and not RSBops:
              # Native sparse
              REFS.append((DDXM_OP, DDZM_OP))
              REFS.append(diffOps1)
       else: 
              # Matrix derivative operators in native format
              REFS.append(diffOps1[0]) # index 12
              REFS.append(diffOps1[1]) # index 13
              
       # Store the terrain profile and operators used on the terrain (diffusion)
       REFS.append(DZT) # index 14
       REFS.append(np.reshape(DZT, (OPS,1), order='F')) # index 15
       REFS.append(sps.csr_array(DDX_BC)) # index 16
       
       # Update REFG with the 2nd vertical derivative of backgrounds
       REFG.append(DDZMS1 @ DQDZ)
       REFG.append(RLOPT[-1])
       
       if not StaticSolve:
              
              # KDtree lookups for filtering sweep
              XMV = np.reshape(XL, (OPS,1), order='F')
              ZMV = np.reshape(ZTL, (OPS,1), order='F')
              XZV = np.hstack((XMV, ZMV))
              
              kdtxz = spk.KDTree(XZV)
              
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
              DZ_wav = 1.0 * abs(DIMS[2]) / (NZ+1)
              print('Uniform grid lengths (m):',DX_wav,DZ_wav)
                     
              dS2 = np.expand_dims(1.0 + np.power(REFS[6][0],2), axis=1)
              dS = np.sqrt(dS2)
              S2 = np.reciprocal(dS2)
              S = np.sqrt(S2)
              DZ = (DIMS[2] - HOPT[0]) / DIMS[2] * DZ_min
              DX = DX_min
                            
              # Smallest physical grid spacing in the 2D mesh
              DLS = min(DX, DZ)
              
              # Compute mesh areas
              DAM = 1.0 / (abs(DIMS[1] - DIMS[0]) * DIMS[2]) * np.reshape(DXM * DZM, (1,OPS), order='F')
              
              # Compute filtering regions
              DL1 = 1.0 * DX_max
              DL2 = 1.0 * DZ_max
              DLR = mt.sqrt(DL1**2 + DL2**2)
              fltDex = kdtxz.query_ball_tree(kdtxz, r=DLR)
              print('Diffusion regions dimensions (m): ', DL1, DL2, DLR)                     

              # Create a container for these quantities
              #DLD = (DL1, DL2, DL1**2, DL2**2, S, dS, DAM, fltDex)
              #'''
              ls = 1.0
              DLD = (np.reshape(ls*DXM, (OPS,), order='F'), 
                     DL2, 
                     np.reshape(np.power(ls*DXM,2), (OPS,), order='F'),
                     DL2**2, S, dS, DAM, fltDex)
              #'''
       # Get memory back
       del(DDXMS1); del(DDZMS1)
       del(DDXMS_CFD); del(DDZMS_CFD)
       del(DDXMS_QS); del(DDZMS_QS)
       del(diffOps1)
       del(PPXMD)
       
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
       
       if isRestart and StaticSolve:
              print('Restarting from previous solution...')
              SOLT, LMS, DCF, NX_in, NZ_in, IT = getFromRestart(restart_file, TOPT, NX, NZ, StaticSolve)
              
              # Updates nolinear boundary condition to next Newton iteration
              dWBC = SOLT[wbdex,0] - dHdX[hdex] * (INIT[ubdex] + SOLT[ubdex,0])  
       else:
              # Set the initial time
              IT = 0.0
              thisTime = IT
              
              # Initial change in vertical velocity at boundary
              dWBC = -dHdX[hdex] * INIT[ubdex]
            
       # Prepare the current fields (TO EVALUATE CURRENT JACOBIAN)
       currentState = np.array(SOLT[:,0])
       fields, U, W = \
              eqs.computePrepareFields(REFS, currentState, INIT, udex, wdex, pdex, tdex)
              
       rhsVec = np.zeros(fields.shape)
       resVec = np.zeros(fields.shape)
       # NetCDF restart for transient runs
       if isRestart and NonLinSolve:
              print('Restarting from: ', fname2Restart)
              try:
                     m_fid = Dataset(fname2Restart, 'r', format="NETCDF4")
                     thisTime = m_fid.variables['time'][rdex]
                     fields[:,0] = np.reshape(m_fid.variables['u'][rdex,:,:,0], (OPS,), order='F')
                     fields[:,1] = np.reshape(m_fid.variables['w'][rdex,:,:,0], (OPS,), order='F')
                     fields[:,2] = np.reshape(m_fid.variables['ln_p'][rdex,:,:,0], (OPS,), order='F')
                     fields[:,3] = np.reshape(m_fid.variables['ln_t'][rdex,:,:,0], (OPS,), order='F')
                     
                     m_fid.close()
                     del(m_fid)
              except:
                     print('Could NOT read restart NC file!', fname2Restart)
       else:
             thisTime = IT
              
       # Initialize output to NetCDF
       hydroState = np.reshape(INIT, (OPS, numVar), order='F')
       newFname = initializeNetCDF(fname4Restart, thisTime, NX, NZ, XL, ZTL, hydroState)
              
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
              rhs, DqDx, DqDz = eqs.computeRHS(fields, hydroState, REFS[10][0], REFS[10][1], REFS[6][0], \
                                                  PHYS, REFS, REFG, ebcDex, zeroDex, True, False, False)
              RHS = np.reshape(rhs, (physDOF,), order='F')
              err = displayResiduals('Current function evaluation residual: ', RHS, \
                                     thisTime, TOPT[0], OPS, udex, wdex, pdex, tdex)
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
                     # Call the 4 way partitioned Schur block solver
                     dsol = dsolver.solveDiskPartSchur(localDir, schurName, f1, f2)
                     
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
              err = displayResiduals(message, RHS, thisTime, TOPT[0], OPS, udex, wdex, pdex, tdex)
              rhs, DqDx, DqDz = eqs.computeRHS(fields, hydroState, REFS[10][0], REFS[10][1], REFS[6][0], \
                                                  PHYS, REFS, REFG, ebcDex, zeroDex, True, False, False)
              RHS = np.reshape(rhs, (physDOF,), order='F')
              message = 'Residual 2-norm AFTER Newton step:'
              err = displayResiduals(message, RHS, thisTime, TOPT[0], OPS, udex, wdex, pdex, tdex)
              del(rhs)
              
              # Check the change in the solution
              DSOL = np.array(SOLT[:,1])
              print('Norm of change in solution: ', np.linalg.norm(DSOL))
       #%% Transient solutions       
       elif NonLinSolve:
              print('Starting Nonlinear Transient Solver...')
              
              # Initialize the perturbations
              if thisTime == 0.0:
                     
                     # Initialize fields
                     if withHydroState:
                            fields[ubdex,1] = -dWBC
                     else:
                            fields[:,0] = hydroState[:,0]
                            fields[ubdex,1] = -dWBC
                            fields[:,2] = hydroState[:,2]
                            fields[:,3] = hydroState[:,3]
                     '''
                     PTR = np.power(np.power((XL + 100.0E3) / 30.0E3, 2.0) + \
                            np.power((ZTL - 10.0E3) / 5.0E3, 2.0), 0.5)
                     PTF = np.power(np.cos(0.5 * mt.pi * PTR), 2.0)
                     PTF = 10.0 * np.where(PTR <= 1.0, PTF, 0.0)
                     
                     LPTF = np.log(1.0 + PTF * np.reciprocal(np.exp(LOGT)) )
                     
                     fields[:,3] = np.reshape(LPTF, (OPS,), order='F')
                     '''
              
              # Initialize fields
              ti = 0; ff = 0
              rhsVec, DqDx, DqDz = eqs.computeRHS(fields, hydroState, REFS[13][0], REFS[13][1], REFS[6][0], \
                                                         PHYS, REFS, REFG, ebcDex, zeroDex, False, False, True)
              fields0 = np.copy(fields)
              rhsVec0 = np.copy(rhsVec)
              resVec0 = np.copy(rhsVec)
              state = fields + hydroState
              error = [np.linalg.norm(rhsVec)]
              
              # Initialize local sound speed and time step
              RdT, T_ratio = eqs.computeRdT(fields, REFS[9][0], PHYS[4])
              VSND = np.sqrt(PHYS[6] * RdT)
              VFLW = np.sqrt(bn.ss(state[:,0:2], axis=1))
              VWAV_max = bn.nanmax(VSND + VFLW)
              TOPT[0] = DTF * DLS / VWAV_max
              print('Initial time step by sound speed: ', str(DLS / VWAV_max) + ' (sec)')
              print('Time stepper order: ', str(TOPT[3]))
              print('Time step factor: ', str(DTF))
              
              OTI = int(TOPT[5] / TOPT[0])
              ITI = int(TOPT[6] / TOPT[0])
              
              # Initialize damping coefficients
              state = fields0 + hydroState
              DCF = rescf.computeResidualViscCoeffs2(DIMS, state, rhsVec, resVec, DLD, ebcDex[2], filteredCoeffs)
              
              while thisTime <= TOPT[4]:
                             
                     # Print out diagnostics every TOPT[5] steps
                     if ti % OTI == 0:
                            
                            message = ''
                            err = displayResiduals(message, np.reshape(rhsVec, (OPS*numVar,), order='F'), \
                                                   thisTime, TOPT[0], OPS, udex, wdex, pdex, tdex)
                            error.append(err)
                            
                            # Store in the NC file
                            store2NC(newFname, thisTime, ff, numVar, NX, NZ, fields, rhsVec, resVec, DCF)
                            
                            ff += 1
                                                 
                     if ti % ITI == 0 and makePlots:
                            makeFieldPlots(TOPT, thisTime, XL, ZTL, fields, rhsVec, resVec, DCF, DCF, NX, NZ, numVar)
                            
                     # Compute the solution within a time step
                     try:   
                            fields, rhsVec, resVec, DCF = tint.computeTimeIntegrationNL(DIMS, PHYS, REFS, REFG, \
                                                                    DLD, TOPT, fields0, rhsVec0, resVec0, hydroState, DCF, \
                                                                    zeroDex, ebcDex, filteredCoeffs, verticalStagger, diffusiveFlux)
                                   
                            ti += 1
                            thisTime += TOPT[0]
                            
                            fields0 = np.copy(fields)
                            rhsVec0 = np.copy(rhsVec)
                            resVec0 = np.copy(resVec)
                            
                            # Compute a time step
                            try: 
                                   # Compute sound speed
                                   RdT, T_ratio = eqs.computeRdT(fields, REFS[9][0], PHYS[4])
                                   VSND = np.sqrt(PHYS[6] * RdT)
                                   
                                   # Compute flow speed
                                   UD = np.abs(state[:,0]); UD[UD < 1.0E-16] = 0.0
                                   WD = np.abs(state[:,1]); WD[WD < 1.0E-16] = 0.0
                                   VFLW = np.sqrt(bn.ss(state[:,0:2], axis=1))
                                   
                                   # Compute new time step based on updated sound speed
                                   VWAV_max = bn.nanmax(VSND + VFLW)
                                   newDT = DTF * DLS / VWAV_max
                                   
                                   #print(newDT)
                                   if not np.isnan(newDT) and not np.isinf(newDT):
                                          TOPT[0] = newDT
                                   
                            except:
                                   print('Bad computation of local sound speed, no change in time step.')
                                   print('Time: ', thisTime, ti)
                            
                     except Exception:
                            print('Transient step failed! Closing out to NC file. Time: ', thisTime)
                            
                            store2NC(newFname, thisTime, ff, numVar, NX, NZ, fields, rhsVec, resVec, DCF)
                            makeFieldPlots(TOPT, thisTime, XL, ZTL, fields, rhsVec, resVec, DCF[0], DCF[1], NX, NZ, numVar)
                            import traceback
                            traceback.print_exc()
                            sys.exit(2)
                     
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
                     plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
              
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
                     plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
                     plt.tight_layout()
              plt.show()
              
              # Check W at the boundaries...
              dwdt = np.reshape(RHS[wdex], (NZ+1, NX+1), order='F')
              return (XL, ZTL, dwdt)
       
if __name__ == '__main__':
       
       #TestName = 'ClassicalSchar01'
       #TestName = 'ClassicalScharIter'
       #TestName = 'UniformStratStatic'
       #TestName = 'DiscreteStratStatic'
       TestName = 'UniformTestTransient'
       #TestName = '3LayerTest'
       
       # Run the model in a loop if needed...
       for ii in range(1):
              diagOutput = runModel(TestName)