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
import os

os.environ["OMP_NUM_THREADS"] = "14" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "14" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "14" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "12" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "12" # export NUMEXPR_NUM_THREADS=6

import sys
import time
import shelve
import shutil
import math as mt
import numpy as np
import numba as nb
import bottleneck as bn
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import matplotlib.path as pth
from matplotlib import cm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Import from the local library of routines
from computeGrid import computeGrid
from computeAdjust4CBC import computeAdjust4CBC
from computeColumnInterp import computeColumnInterp
from computeTopographyOnGrid import computeTopographyOnGrid
from computeTemperatureProfileOnGrid import computeTemperatureProfileOnGrid
from computeThermoMassFields import computeThermoMassFields
from computeShearProfileOnGrid import computeShearProfileOnGrid
from computeRayleighEquations import computeRayleighEquations

# Numerical stuff
import computePartialDerivativesXZ as devop
import computeGuellrichDomain2D as coords
import computeDiskPartSchur as dsolver
import computeDerivativeMatrix as derv
import computeEulerEquationsLogPLogT as eqs
import computeTimeIntegration as tint

import faulthandler; faulthandler.enable()

from netCDF4 import Dataset

# Disk settings
localDir = '/home/jeguerra/scratch/'
restart_file = localDir + 'restartDB'
schurName = localDir + 'SchurOps'
fname2Restart = '/media/jeguerra/DataBuffer/Simulation2Restart.nc'
fname4Restart = '/media/jeguerra/DataBuffer/SimulationTemp.nc'       

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

def displayResiduals(message, rhs, thisTime, DT, OPS):
       RHSA = np.abs(rhs)
       err = bn.nanmax(RHSA, axis=0)# / mt.sqrt(OPS)
       if message != '':
              print(message)
       print('Time (min): %4.2f, DT (sec): %4.3E, Residuals: %10.4E, %10.4E, %10.4E, %10.4E' \
             % (thisTime / 60.0, DT, err[0], err[1], err[2], err[3]))
       
       return err

def initializeNetCDF(fname, thisTime, XL, ZTL, hydroState, senseTemp):
       
       NX = XL.shape[1]
       NZ = ZTL.shape[0]
       
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
              os.remove(fname)
              m_fid = Dataset(newFname, 'w', format="NETCDF4")
              
       # Make dimensions
       m_fid.createDimension('time', None)
       m_fid.createDimension('x', NX)
       m_fid.createDimension('y', 1)
       m_fid.createDimension('z', NZ)
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
       GVAR = m_fid.createVariable('TZ', 'f8', ('z', 'x', 'y'))
       PVAR = m_fid.createVariable('LNP', 'f8', ('z', 'x', 'y'))
       TVAR = m_fid.createVariable('LNT', 'f8', ('z', 'x', 'y'))
       # Store variables
       UVAR[:,:,0] = np.reshape(hydroState[:,0], (NZ,NX), order='F')
       GVAR[:,:,0] = np.reshape(senseTemp, (NZ,NX), order='F')
       PVAR[:,:,0] = np.reshape(hydroState[:,2], (NZ,NX), order='F')
       TVAR[:,:,0] = np.reshape(hydroState[:,3], (NZ,NX), order='F')
       # Create variables (transient fields)
       m_fid.createVariable('u', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('w', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('ln_p', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('ln_t', 'f8', ('time', 'z', 'x', 'y'))
       #'''
       # Create variables (field dynamical tendencies)
       m_fid.createVariable('DuDt', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('DwDt', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Dln_pDt', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('Dln_tDt', 'f8', ('time', 'z', 'x', 'y'))
       #'''
       #'''
       # Create variables (field DynSGS tendencies)
       m_fid.createVariable('SGSu', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('SGSw', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('SGSln_p', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('SGSln_t', 'f8', ('time', 'z', 'x', 'y'))
       #'''
       '''
       # Create variables for diffusion coefficients
       m_fid.createVariable('DC1', 'f8', ('time', 'z', 'x', 'y'))
       m_fid.createVariable('DC2', 'f8', ('time', 'z', 'x', 'y'))
       '''
       m_fid.close()
       del(m_fid)
       
       return newFname

def store2NC(newFname, thisTime, ff, numVar, ZTL, fields, rhsVec, resVec):
       
       NX = ZTL.shape[1]
       NZ = ZTL.shape[0]
       
       # Store in the NC file
       try:
              m_fid = Dataset(newFname, 'a', format="NETCDF4")
              m_fid.variables['time'][ff] = thisTime
              
              for pp in range(numVar):
                     q = np.reshape(fields[:,pp], (NZ, NX), order='F')
                     dqdt = np.reshape(rhsVec[:,pp], (NZ, NX), order='F')
                     rq = np.reshape(resVec[:,pp], (NZ, NX), order='F')

                     if pp == 0:
                            m_fid.variables['u'][ff,:,:,0] = q
                            m_fid.variables['DuDt'][ff,:,:,0] = dqdt
                            m_fid.variables['SGSu'][ff,:,:,0] = rq
                     elif pp == 1:
                            m_fid.variables['w'][ff,:,:,0] = q
                            m_fid.variables['DwDt'][ff,:,:,0] = dqdt
                            m_fid.variables['SGSw'][ff,:,:,0] = rq
                     elif pp == 2:
                            m_fid.variables['ln_p'][ff,:,:,0] = q
                            m_fid.variables['Dln_pDt'][ff,:,:,0] = dqdt
                            m_fid.variables['SGSln_p'][ff,:,:,0] = rq
                     else:
                            m_fid.variables['ln_t'][ff,:,:,0] = q
                            m_fid.variables['Dln_tDt'][ff,:,:,0] = dqdt
                            m_fid.variables['SGSln_t'][ff,:,:,0] = rq
                            
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
       
       # Set the solution type
       StaticSolve = thisTest.solType['StaticSolve']
       NewtonLin = thisTest.solType['NewtonLin']
       ExactBC = thisTest.solType['ExactBC']
       
       # Set the grid type
       HermFunc = thisTest.solType['HermFuncGrid']
       VertSpec = thisTest.solType['VerticalSpectral']
       
       useGuellrich = True
       useUniformSt = False
       
       # Switch to use the PyRSB multithreading module (CPU multithreaded SpMV)
       if StaticSolve:
              RSBops = False
       else:
              RSBops = False # Turn off PyRSB SpMV
              if RSBops:
                     from rsb import rsb_matrix
              else:
                     import torch as tch
                     if tch.cuda.is_available():
                            machine = tch.device('cuda')
                            print("GPU is available")
                     else:
                            machine = tch.device('cpu')
                            print("GPU not available, using CPU instead")
       
       # Use the uniform grid fourier solution if not Hermite Functions
       if HermFunc:
              FourierLin = False
              print('Hermite Function grid in the horizontal.')
       else:
              FourierLin = False
              if FourierLin == True:
                     print('Uniform Fourier grid in the horizontal.')
              else:
                     print('Uniform INTERIOR grid in the horizontal')
                     
       if VertSpec:
              verticalChebGrid = False
              verticalLegdGrid = True
       else:
              verticalChebGrid = False
              verticalLegdGrid = False
       
       if verticalChebGrid:
              print('Chebyshev spectral derivative in the vertical.')
       elif verticalLegdGrid:
              print('Legendre spectral derivative in the vertical.')
       else:
              print('Uniform INTERIOR grid in the vertical')
              
       # Set residual diffusion switch
       DynSGS_RES = True
       if DynSGS_RES:
              print('Diffusion coefficients by RES estimate.')
       else:
              print('Diffusion coefficients by RHS evaluation.')
       
       # Set direct solution method
       SolveSchur = thisTest.solType['SolveSchur']
       
       # Set Newton solve initial and restarting parameters
       isRestart = thisTest.solType['IsRestart'] # Initializes from a restart database
       
       if isRestart:
              rdex = -1
       
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
       numVar = len(varDex)
       
       Z_in = thisTest.Z_in
       T_in = thisTest.T_in
              
       #%% COMPUTE STRATIFICATION AT HIGH RESOLUTION SPECTRAL
       chebHydro = False
       legrHydro = True
       DIM0 = [DIMS[0], DIMS[1], DIMS[2], DIMS[3], 255, DIMS[5]]
       REF0 = computeGrid(DIM0, HermFunc, FourierLin, chebHydro, legrHydro)
       
       DDXP, dummy = derv.computeCubicSplineDerivativeMatrix(REF0[0], False, True, None)
       
       # Get the spectral resolution operator here
       if chebHydro:
              DDZP, ITRANS = derv.computeChebyshevDerivativeMatrix(DIM0)
       elif legrHydro:
              DDZP, ITRANS = derv.computeLegendreDerivativeMatrix(DIM0)
       else:
              DDZP = derv.computeCompactFiniteDiffDerivativeMatrix1(REF0[1], 6)
                     
       REF0.append(DDXP)
       REF0.append(DDZP)
       
       hx, dhx = computeTopographyOnGrid(REF0, HOPT)
       zRay = DIMS[2] - RLOPT[0]
       
       if useGuellrich:
              xl, ztl, dzt, sig, ZRL = \
                     coords.computeGuellrichDomain2D(DIM0, REF0[0], REF0[1], zRay, hx, dhx, StaticSolve)
       
       if useUniformSt:
              xl, ztl, dzt, sig, ZRL = \
                     coords.computeStretchedDomain2D(DIM0, REF0, zRay, hx, dhx)
       
       REF0.append(xl)
       REF0.append(ztl)
       REF0.append(dhx)
       REF0.append(sig)
       
       tz, dtz = \
              computeTemperatureProfileOnGrid(PHYS, REF0, Z_in, T_in, smooth3Layer, uniformStrat)
              
       dlpz, lpz, pz, dlptz, lpt, pt, rho = \
              computeThermoMassFields(PHYS, DIM0, REF0, tz, dtz, 'sensible', RLOPT)
              
       uz, duz = computeShearProfileOnGrid(REF0, JETOPS, PHYS[1], pz, dlpz, uniformWind, linearShear)
       
       #%% SET UP THE GRID AND INDEX VECTORS
       zRay = DIMS[2] - RLOPT[0]
       REFS = computeGrid(DIMS, HermFunc, FourierLin, verticalChebGrid, verticalLegdGrid)
       
       # Update OPS to the actual grid
       NX = REFS[0].shape[0]
       NZ = REFS[1].shape[0]
       OPS = NX * NZ
       physDOF = numVar * OPS
       
       #%% Read in topography profile or compute from analytical function
       HofX, dHdX = computeTopographyOnGrid(REFS, HOPT)
       
       if useGuellrich:
              XL, ZTL, DZT, sigma, ZRL = \
                     coords.computeGuellrichDomain2D(DIMS, REFS[0], REFS[1], zRay, HofX, dHdX, StaticSolve)
       
       if useUniformSt:
              XL, ZTL, DZT, sigma, ZRL = \
                     coords.computeStretchedDomain2D(DIMS, REFS, zRay, HofX, dHdX)
       
       #% Compute the BC index vector
       ubcDex, wbcDex, pbcDex, tbcDex, sysDex, ebcDex = \
              computeAdjust4CBC(ZTL.shape, numVar, varDex, bcType)
      
       #%% Compute the raw derivative matrix operators in alpha-xi computational space
       if HermFunc and not FourierLin:
              DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       elif FourierLin and not HermFunc:
              DDX_1D, HF_TRANS = derv.computeFourierDerivativeMatrix(DIMS)
       else:
              print('NON-SPECTRAL derivative in the horizontal.')
              DDX_BC = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 8)
              DDX_1D, DDX4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[0], True, False, DDX_BC)
                            
       if verticalChebGrid and not verticalLegdGrid:
              DDZ_1D, VTRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
       elif verticalLegdGrid and not verticalChebGrid:
              DDZ_1D, VTRANS = derv.computeLegendreDerivativeMatrix(DIMS)
       else:
              print('NON-SPECTRAL derivative in the verticals.')
              DDZ_BC = derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 8)
              DDZ_1D, DDX4_QS = derv.computeQuinticSplineDerivativeMatrix(REFS[1], True, False, DDZ_BC)
       
       # Derivative operators
       useAverageDerivatives = False
       DDXL, dummy = derv.computeCubicSplineDerivativeMatrix(REFS[0], False, True, None)
       DDZL, dummy = derv.computeCubicSplineDerivativeMatrix(REFS[1], False, True, None)
       DDXH, dummy = derv.computeQuinticSplineDerivativeMatrix(REFS[0], False, True, DDX_BC)
       DDZH, dummy = derv.computeQuinticSplineDerivativeMatrix(REFS[1], False, True, DDZ_BC)
       
       if useAverageDerivatives:
              print('Average spline and compact FD derivatives.')
              DDXL += derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 6)
              DDZL += derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 6)
              DDXH += derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[0], 8)
              DDZH += derv.computeCompactFiniteDiffDerivativeMatrix1(REFS[1], 8)
              DDXL *= 0.5; DDZL *= 0.5; DDXH *= 0.5; DDZH *= 0.5
       
       # Derivative operators from global spectral methods
       DDXMS1, DDZMS1 = devop.computePartialDerivativesXZ(DIMS, sigma, DDX_1D, DDZ_1D)
       DDXMS_LO, DDZMS_LO = devop.computePartialDerivativesXZ(DIMS, sigma, 
                                                              DDXL, DDZL)
       DDXMS_HO, DDZMS_HO = devop.computePartialDerivativesXZ(DIMS, sigma, 
                                                              DDXH, DDZH)
       
       #%% MAKE THE INITIAL/BACKGROUND STATE ON COMPUTATIONAL GRID
       
       # Compute the initial fields by interolation
       '''
       import scipy.interpolate as spi
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
       #'''
       # Compute the background (initial) fields
       TZ = np.expand_dims(tz, axis=1)
       TZ = computeColumnInterp(DIM0, REF0[1], TZ, ZTL, ITRANS)
       DTDZ = np.expand_dims(dtz, axis=1)
       DTDZ = computeColumnInterp(DIM0, REF0[1], DTDZ, ZTL, ITRANS)
       U = np.expand_dims(uz, axis=1)
       UZ = computeColumnInterp(DIM0, REF0[1], U, ZTL, ITRANS)
       dUdz = np.expand_dims(duz, axis=1)
       DUDZ = computeColumnInterp(DIM0, REF0[1], dUdz, ZTL, ITRANS)
       LPZ = np.expand_dims(lpz, axis=1)
       LOGP = computeColumnInterp(DIM0, REF0[1], LPZ, ZTL, ITRANS)
       LPT = np.expand_dims(lpt, axis=1)
       LOGT = computeColumnInterp(DIM0, REF0[1], LPT, ZTL, ITRANS)       
       #'''
       # Compute thermodynamic gradients (no interpolation!)
       PORZ = PHYS[3] * TZ
       PBAR = np.exp(LOGP) # Hydrostatic pressure
       DLPDZ = -PHYS[0] / (PHYS[3] * TZ)
       DLTDZ = np.reciprocal(TZ) * DTDZ
       DLPTDZ = DLTDZ - PHYS[4] * DLPDZ

       # Get the static vertical gradients and store
       DUDZ = np.reshape(DUDZ, (OPS,1), order='F')
       DLTDZ = np.reshape(DLTDZ, (OPS,1), order='F')
       DLPDZ = np.reshape(DLPDZ, (OPS,1), order='F')
       DLPTDZ = np.reshape(DLPTDZ, (OPS,1), order='F')
       DQDZ = np.hstack((DUDZ, np.zeros((OPS,1)), DLPDZ, DLPTDZ))
       
       if StaticSolve:
              UZ += JETOPS[0]
       else:
              UZ += 0.0
       
       #%% RAYLEIGH AND GML WEIGHT OPERATORS
       RLM, GML = computeRayleighEquations(DIMS, XL, ZTL, ZRL, RLOPT)
       
       # Make a collection for background field derivatives
       REFG = [RLM, GML, DLTDZ, DQDZ, RLOPT[4]]
                     
       #%% PREPARE DERIVATIVE OPERATORS
       if HermFunc:
              PPXMA = DDXMS1
       else:
              PPXMA = DDXMS_HO
              
       if verticalChebGrid or verticalLegdGrid:
              #advtOps = (REFG[0][1].dot(PPXMD),
              #           REFG[0][2].dot(DDZMS1))
              advtOps = (PPXMA,DDZMS1)
       else:
              #advtOps = (REFG[0][1].dot(PPXMD),
              #           REFG[0][2].dot(DDZMS_HO))
              advtOps = (PPXMA,DDZMS_HO)
       
       diffOps = (DDXMS_HO,DDZMS_HO)
       
       # Update the REFS collection
       REFS.append(DDX_1D) # index 2
       REFS.append(DDZ_1D) # index 3
       REFS.append(XL) # index 4
       REFS.append(ZTL) # index 5
       REFS.append((dHdX, HofX)) # index 6
       REFS.append(sigma) # index 7
       REFS.append(np.reshape(UZ, (OPS,), order='F')) # index 8
       REFS.append([np.reshape(PORZ, (OPS,), order='F'), 
                    np.reshape(PBAR, (OPS,), order='F')]) #index 9
       REFS.append(advtOps) # index 10
       REFS.append(diffOps) # index 11
       
       #%% Store operators for use
       if StaticSolve:
              # Native sparse
              DDOP = sps.vstack(advtOps, format='csr')
              REFS.append(DDOP) # index 12
              DDOP = sps.vstack(diffOps, format='csr')
              REFS.append(DDOP) # index 13
       else:
              # Multithreaded enabled for transient solution
              if RSBops:
                     DDOP = sps.vstack(advtOps, format='csr')
                     DDOP = rsb_matrix(sps.csr_array(DDOP), shape=DDOP.shape)
                     DDOP.autotune(nrhs=numVar)
                     REFS.append(DDOP) # index 12
                     
                     DDOP = sps.vstack(diffOps, format='csr')
                     DDOP = rsb_matrix(sps.csr_array(DDOP), shape=DDOP.shape)
                     DDOP.autotune(nrhs=2*numVar)
                     REFS.append(DDOP) # index 13
              else:
                     DDOP = sps.vstack(advtOps, format='coo') 
                     ind = np.vstack((DDOP.row, DDOP.col))
                     val = DDOP.data
                     DDOP = tch.sparse_coo_tensor(ind, val, device=machine)
                     DDOP = DDOP.to_sparse_csr()
                     
                     REFS.append(DDOP) # index 12
                     del(DDOP)
                     
                     DDOP = sps.vstack(diffOps, format='coo')
                     ind = np.vstack((DDOP.row, DDOP.col))
                     val = DDOP.data
                     DDOP = tch.sparse_coo_tensor(ind, val, device=machine)
                     DDOP = DDOP.to_sparse_csr()
                     
                     REFS.append(DDOP) # index 13
                     del(val)
                     del(ind)
              
       # Store the terrain profile and operators used on the terrain (diffusion)
       REFS.append(np.reshape(DZT, (OPS,1), order='F')) # index 14
       
       dS2 = np.expand_dims(1.0 + np.power(dHdX,2), axis=1)
       S2 = np.reciprocal(dS2)
       S = np.sqrt(S2)
       REFS.append(S)
              
       #%% Get some memory back here
       del(PORZ)
       del(DUDZ)
       del(DLTDZ)
       del(DLPDZ)
       del(DLPTDZ)
       del(GML)
       del(REF0)
       del(DIM0)
       del(DDOP)
       del(advtOps); del(diffOps)
       del(DDXMS1); del(DDZMS1)
       del(DDXMS_LO); del(DDZMS_LO)
       del(DDXMS_HO); del(DDZMS_HO)
       del(dummy)
       
       #%% SOLUTION INITIALIZATION

       hydroState = np.empty((OPS,numVar))
       fields = np.zeros((OPS,numVar))
       state = np.empty((OPS,numVar))
       dfields = np.empty((OPS,numVar))
       rhsVec = np.empty((OPS,numVar))
       rhsSGS = np.empty((OPS,numVar))
       resVec = np.empty((OPS,numVar))
       
       # Initialize residual coefficient storage
       dcf = np.zeros((fields.shape[0],2,1))
       
       if isRestart:
              print('Restarting from: ', fname2Restart)
              try:
                     m_fid = Dataset(fname2Restart, 'r', format="NETCDF4")
                     thisTime = m_fid.variables['time'][rdex]
                     
                     hydroState[:,0] = np.reshape(m_fid.variables['U'][:,:,0], (OPS,), order='F')
                     hydroState[:,1] = 0.0
                     hydroState[:,2] = np.reshape(m_fid.variables['LNP'][:,:,0], (OPS,), order='F')
                     hydroState[:,3] = np.reshape(m_fid.variables['LNT'][:,:,0], (OPS,), order='F')
                     
                     state[:,0] = np.reshape(m_fid.variables['u'][rdex,:,:,0], (OPS,), order='F')
                     state[:,1] = np.reshape(m_fid.variables['w'][rdex,:,:,0], (OPS,), order='F')
                     state[:,2] = np.reshape(m_fid.variables['ln_p'][rdex,:,:,0], (OPS,), order='F')
                     state[:,3] = np.reshape(m_fid.variables['ln_t'][rdex,:,:,0], (OPS,), order='F')
                     
                     fields = state - hydroState
                     
                     rhsVec[:,0] = np.reshape(m_fid.variables['DuDt'][rdex,:,:,0], (OPS,), order='F')
                     rhsVec[:,1] = np.reshape(m_fid.variables['DwDt'][rdex,:,:,0], (OPS,), order='F')
                     rhsVec[:,2] = np.reshape(m_fid.variables['Dln_pDt'][rdex,:,:,0], (OPS,), order='F')
                     rhsVec[:,3] = np.reshape(m_fid.variables['Dln_tDt'][rdex,:,:,0], (OPS,), order='F')
                     
                     resVec[:,0] = np.reshape(m_fid.variables['SGSu'][rdex,:,:,0], (OPS,), order='F')
                     resVec[:,1] = np.reshape(m_fid.variables['SGSw'][rdex,:,:,0], (OPS,), order='F')
                     resVec[:,2] = np.reshape(m_fid.variables['SGSln_p'][rdex,:,:,0], (OPS,), order='F')
                     resVec[:,3] = np.reshape(m_fid.variables['SGSln_t'][rdex,:,:,0], (OPS,), order='F')
                     
                     m_fid.close()
                     del(m_fid)
              except Exception as e:
                     print('Could NOT read restart NC file!', fname2Restart)
                     print(e)
              isInitialStep = False
       else:
              hydroState[:,0] = np.reshape(UZ, (OPS,), order='F')
              hydroState[:,1] = 0.0
              hydroState[:,2] = np.reshape(LOGP, (OPS,), order='F')
              hydroState[:,3] = np.reshape(LOGT, (OPS,), order='F')
              
              state = fields + hydroState
              
              # Set the initial time
              IT = 0.0
              thisTime = IT
              isInitialStep = True
              
       # Compute the terrain boundary condition
       dUBC = state[ebcDex[2],0]
       dWBC = fields[ebcDex[2],1] - dHdX * dUBC
              
       # Initialize output to NetCDF
       newFname = initializeNetCDF(fname4Restart, thisTime, XL, ZTL, hydroState, TZ)
              
       start = time.time()
       
       #%% STATIC OR TRANSIENT SOLVERS
       if StaticSolve:
              
              # Initialize solution storage
              SOLT = np.zeros((physDOF, 2))
              SOLT[:,0] = np.reshape(fields, (physDOF,), order='F')
              
              # Initialize Lagrange Multiplier storage (bottom boundary)
              LMS = np.zeros(NX)
                     
              # Compute the RHS for this iteration
              rhsVec, DqDx, DqDz = eqs.computeRHS(state, fields, REFS[10][0], REFS[10][1], \
                                                  PHYS, REFS, REFG, True, False)
              displayResiduals('Current function evaluation residual: ', rhsVec, thisTime, TOPT[0], OPS)
              
              if NewtonLin:
                     # Full Newton linearization with TF terms
                     DOPS_NL = eqs.computeJacobianMatrixLogPLogT(PHYS, REFS, REFG, \
                                   fields, state[:,0], ebcDex[2], ebcDex[3])
              else:
                     # Classic linearization without TF terms
                     DOPS_NL = eqs.computeEulerEquationsLogPLogT_Classical(DIMS, PHYS, 
                                                                           REFS, REFG)

              print('Compute Jacobian operator blocks: DONE!')
              
              # Convert blocks to 'lil' format for efficient indexing
              DOPS = []
              for dd in range(len(DOPS_NL)):
                     if (DOPS_NL[dd]) is not None:
                            DOPS.append(DOPS_NL[dd].tolil())
                     else:
                            DOPS.append(DOPS_NL[dd])
              del(DOPS_NL)
              
              # Compute Lagrange Multiplier column augmentation matrices (terrain equation)
              C1 = -1.0 * sps.diags(dHdX, offsets=0, format='csr')
              C2 = +1.0 * sps.eye(NX, format='csr')
       
              colShape = (OPS,NX)
              LD = sps.lil_array(colShape)
              if ExactBC:
                     LD[ebcDex[2],:] = C1
              LH = sps.lil_array(colShape)
              LH[ebcDex[2],:] = C2
              LM = sps.lil_array(colShape)
              LQ = sps.lil_array(colShape)
              
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
              LDIA = sps.lil_array((NX,NX))
              
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
              ROPS = sps.spdiags(RLOPT[4] * RLM[-1][:,0], 0, OPS, OPS).tolil()
              A += ROPS[np.ix_(ubcDex,ubcDex)]
              F += ROPS[np.ix_(wbcDex,wbcDex)]
              K += ROPS[np.ix_(pbcDex,pbcDex)]
              Q += ROPS[np.ix_(tbcDex,tbcDex)]
              
              del(DOPS)
              
              # Set up Schur blocks or full operator...
              if SolveSchur:
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
                     fu = rhsVec[:,0]
                     fw = rhsVec[:,1]
                     f1 = np.concatenate((-dWBC, fu[ubcDex], fw[wbcDex]))
                     fp = rhsVec[:,2]
                     ft = rhsVec[:,3]
                     f2 = np.concatenate((fp[pbcDex], ft[tbcDex]))   
              else:
                     # Compute the global linear operator
                     
                     AN = sps.bmat([[LDIA, LNA, LOA, LPA, LQAR], \
                              [LDA, A, B, C, D], \
                              [LHA, E, F, G, H], \
                              [LMA, I, J, K, M], \
                              [LQAC, N, O, P, Q]], format='csc')
              
                     # Compute the global linear force vector
                     RHS = np.reshape(rhsVec, (physDOF,), order='F')
                     bN = np.concatenate((-dWBC, RHS[sysDex]))
              
              # Get memory back
              del(A); del(B); del(C); del(D)
              del(E); del(F); del(G); del(H)
              del(I); del(J); del(K); del(M)
              del(N); del(O); del(P); del(Q)
              print('Set up global linear operators: DONE!')
       
              print('Starting Linear to Nonlinear Static Solver...')
              
              if SolveSchur:
                     # Call the 4 way partitioned Schur block solver
                     dsol = dsolver.solveDiskPartSchur(localDir, schurName, f1, f2)
              else:
                     print('Solving linear system by full operator SuperLU...')
                     # Direct solution over the entire operator (better for testing BC's)
                     opts = dict(Equil=True, IterRefine='DOUBLE')
                     factor = spl.splu(AN, permc_spec='MMD_ATA', options=opts)
                     del(AN)
                     dsol = factor.solve(bN)
                     del(bN)
                     del(factor)
                     
              # Store the Lagrange Multipliers
              LMS += dsol[0:NX]
              dsolQ = dsol[NX:]
              
              SOLT[sysDex,0] += dsolQ

              # Store solution change to instance 1
              SOLT[sysDex,1] = dsolQ
              
              print('Recover full linear solution vector... DONE!')
              
              # Prepare the fields
              fields[:] = np.reshape(SOLT[:,0], (OPS,numVar), order='F')
              state[:] = fields + hydroState
              
              #%% Set the output residual and check
              message = 'Residual 2-norm BEFORE Newton step:'
              displayResiduals(message, rhsVec, thisTime, TOPT[0], OPS)
              try:
                     rhsVec, DqDx, DqDz = eqs.computeRHS(state, fields, REFS[10][0], REFS[10][1], \
                                                  PHYS, REFS, REFG, True, False)
              except:
                     rhsVec = np.zeros(fields.shape)
                     
              message = 'Residual 2-norm AFTER Newton step:'
              displayResiduals(message, rhsVec, thisTime, TOPT[0], OPS)
              
              store2NC(newFname, thisTime, 0, numVar, ZTL, fields, rhsVec, resVec, dcf)
       
       # Transient solution
       else:
              print('Starting Nonlinear Transient Solver...')
              
              # Forward differences
              DXV1 = np.diff(XL, axis=1, append=np.expand_dims(XL[:,-2],axis=1))
              DZV1 = np.diff(ZTL, axis=0, append=np.expand_dims(ZTL[-2,:],axis=0))
              # Backward differences
              DXV2 = np.diff(np.flip(XL, axis=1), axis=1, append=np.expand_dims(XL[:,1],axis=1))
              DZV2 = np.diff(np.flip(ZTL, axis=0), axis=0, append=np.expand_dims(ZTL[1,:],axis=0))
              # Average F-B differences to center at nodes
              DXV = 0.5 * (np.abs(DXV1) + np.abs(DXV2))
              DZV = 0.5 * (np.abs(DZV1) + np.abs(DZV2))
              
              # Estimate grid point areas
              DA = np.reshape(np.abs(DXV * DZV), (OPS,), order='F')
              
              # Compute DX and DZ grid length scales
              DX_min = 1.0 * bn.nanmin(DXV)
              DZ_min = 1.0 * bn.nanmin(DZV)
              print('Minimum grid lengths:',DX_min,DZ_min)
              DX_avg = 1.0 * bn.nanmean(DXV)
              DZ_avg = 1.0 * bn.nanmean(DZV)
              print('Average grid lengths:',DX_avg,DZ_avg)
              DX_max = 1.0 * bn.nanmax(DXV)
              DZ_max = 1.0 * bn.nanmax(DZV)
              print('Maximum grid lengths:',DX_max,DZ_max)
                            
              # Smallest physical grid spacing in the 2D mesh
              DLS = min(DX_min, DZ_min)
              
              # Compute filtering regions by KDtree lookups
              XMV = np.reshape(XL, (OPS,1), order='F')
              ZMV = np.reshape(ZTL, (OPS,1), order='F')
              XZV = np.hstack((XMV, ZMV))
              
              # DynSGS filter scale lengths
              D_fil1 = 2.0 * DX_max
              D_fil2 = 2.0 * DZ_max
              D_mag1 = 2.0 * D_fil1
              D_mag2 = 2.0 * D_fil2
              DLR = mt.sqrt(D_fil1**2 + D_fil2**2)
              #DLR = 0.5 * (DL1 + DL2)
              
              isRectRegion = True
              def searchRegions(nn):
                     node = XZV[nn,:]
                     
                     if isRectRegion:
                            #'''
                            if np.any(XL[:,0] == node[0]) or np.any(XL[:,-1] == node[0]):
                                   DLO = 2*D_fil1
                                   verts = np.array([(node[0] + DLO, node[1] - D_fil2), \
                                            (node[0] + DLO, node[1] + D_fil2), \
                                            (node[0] - DLO, node[1] + D_fil2), \
                                            (node[0] - DLO, node[1] - D_fil2)])
                                                                      
                            elif np.any(ZTL[0,:] == node[1]) or np.any(ZTL[-1,:] == node[1]):
                                   DLO = 2*D_fil2
                                   verts = np.array([(node[0] + D_fil1, node[1] - DLO), \
                                            (node[0] + D_fil1, node[1] + DLO), \
                                            (node[0] - D_fil1, node[1] + DLO), \
                                            (node[0] - D_fil1, node[1] - DLO)])
                                   
                            else:
                                   verts = np.array([(node[0] + D_fil1, node[1] - D_fil2), \
                                            (node[0] + D_fil1, node[1] + D_fil2), \
                                            (node[0] - D_fil1, node[1] + D_fil2), \
                                            (node[0] - D_fil1, node[1] - D_fil2)])
              
                            region = pth.Path(verts).contains_points(XZV)
                     else:
                            circle = pth.Path.circle(center=(node[0],node[1]),
                                                     radius=DLR)
                            region = circle.contains_points(XZV)
                     
                     regDex = np.nonzero(region == True)[0].tolist()
                     
                     return regDex
              #'''
              ncores = os.cpu_count() - 2
              fltDex = Parallel(n_jobs=ncores)(delayed(searchRegions)(ii) \
                                          for ii in np.arange(XZV.shape[0]))
              #'''
              #fltDex = [searchRegions(ii) for ii in np.arange(XZV.shape[0])]
              
              def meanRegions(regDex):
                     # Function mean kernel
                     mkernel = DA[regDex] / bn.nansum(DA[regDex])                     
                     
                     return mkernel
              
              fltDms = [meanRegions(regDex) for regDex in fltDex]

              # Manipulate arrays to enable numba acceleration for DynSGS
              nb_list = nb.typed.List
              regDex = nb_list(np.array(dex, dtype=np.int32) for dex in fltDex)
              regDms = nb_list(np.array(dms, dtype=np.float64) for dms in fltDms)

              # Create a container for DynSGS scaling and region parameters
              DLD = (D_mag1, D_mag2, D_mag1**2, D_mag2**2,
                     DTF * DLS, S, DA / DIMS[-1], regDex, regDms)
              
              print('Diffusion regions dimensions (m): ', D_fil1, D_fil2)
              print('Diffusion magnitude scale (m): ', D_mag1, D_mag2)
              
              # Compute sound speed and initial time step
              RdT, T_ratio = eqs.computeRdT(PHYS, state, fields, REFS[9][0])
              TOPT[0], VWAV_ref = eqs.computeNewTimeStep(PHYS, RdT, state, 
                                                                   DLD, isInitial=isInitialStep)
              print('Initial Sound Speed (m/s): ', VWAV_ref)
              
              # Normalization for vertical velocity
                     
              # compute function average of initial fields
              sol_avrg = DLD[-3] @ state
              # compute state relative to average
              res_norm = DLD[-3] @ np.abs(state - sol_avrg)
              
              # Fix the kinematic norms
              rw = np.abs(dWBC)
              res_norm[1] = bn.nanmean(rw[rw > 0.0])
              
              print('Initial Residual Norms:')
              res_norm = np.diag(1.0 / res_norm)
              print(res_norm)
       
              # Initialize parameters
              ti = 0; ff = 0
              print('Time stepper order: ', TOPT[3])
              print('Initial time step:', str(TOPT[0]))
              
              # Output time global accumulators
              diagTime = 0.0
              
              # Impose BC's on all states
              hydroState = eqs.enforceBC_SOL(hydroState, ebcDex, hydroState)
              state = eqs.enforceBC_SOL(state, ebcDex, hydroState)
              
              if RSBops:
                     thisDt = TOPT[0]
              else:
                     PHYS = [tch.tensor(item, dtype=tch.double, device=machine) 
                             for item in PHYS]
                     state = tch.tensor(state, dtype=tch.double, device=machine)
                     hydroState = tch.tensor(hydroState, dtype=tch.double, device=machine)
                     dfields = tch.tensor(dfields, dtype=tch.double, device=machine)
                     rhsVec = tch.tensor(rhsVec, dtype=tch.double, device=machine)
                     res_norm = tch.tensor(res_norm, dtype=tch.double, device=machine)
                     dcf = tch.tensor(dcf, dtype=tch.double, device=machine)
                     thisDt = tch.tensor(TOPT[0], dtype=tch.double, device=machine)
                     thisTime = tch.tensor(thisTime, dtype=tch.double, device=machine)
                     
                     REFG[0][:-1] = [tch.tensor(item, dtype=tch.double, device=machine) 
                                     for item in REFG[0][:-1]]
                     
                     REFS[9] = [tch.tensor(item, dtype=tch.double, device=machine) 
                                for item in REFS[9]]
                     
                     REFG[3] = tch.tensor(REFG[3], dtype=tch.double, device=machine)
                     REFS[14] = tch.tensor(REFS[14], dtype=tch.double, device=machine)
                     REFS[15] = tch.tensor(REFS[15], dtype=tch.double, device=machine)
              
              while thisTime <= TOPT[4]:
                            
                     # Compute the solution within a time step
                     try:   
                            state, dfields, \
                            rhsVec, rhsSGS, resVec, \
                            dcf, thisDt = tint.computeTimeIntegrationNL(PHYS, REFS, REFG,
                                                                    DLD, thisDt, TOPT, 
                                                                    state, hydroState, rhsVec, 
                                                                    dfields, dcf, ebcDex, RSBops,
                                                                    VWAV_ref, res_norm, 
                                                                    isInitialStep, DynSGS_RES)
                            
                     except Exception:
                            print('Transient step failed! Closing out to NC file. Time: ', thisTime)
                            
                            store2NC(newFname, thisTime, ff, 
                                     numVar, NX, NZ, 
                                     state, rhsVec, rhsSGS)
                            makeFieldPlots(TOPT, thisTime, XL, ZTL, state, rhsVec, rhsSGS, dcf[0], dcf[1], NX, NZ, numVar)
                            import traceback
                            traceback.print_exc()
                            sys.exit(2)
                            
                     # Print out diagnostics every TOPT[5] seconds
                     if diagTime >= TOPT[5] or ti == 0:
                            
                            message = ''
                                   
                            # Store in the NC file
                            if RSBops:
                                   displayResiduals(message, rhsVec, 
                                                    thisTime, thisDt, OPS)
                                   store2NC(newFname, thisTime, 
                                            ff, numVar, ZTL, 
                                            state, rhsVec, rhsSGS)
                            else:
                                   displayResiduals(message, rhsVec.cpu().numpy(), 
                                                    thisTime.cpu().numpy(), 
                                                    thisDt.cpu().numpy(), OPS)
                                   store2NC(newFname, thisTime.cpu().numpy(), 
                                            ff, numVar, ZTL, 
                                            state.cpu().numpy(), 
                                            rhsVec.cpu().numpy(), 
                                            rhsSGS.cpu().numpy())
                                                        
                            ff += 1
                            diagTime = 0.0
                            
                            if isInitialStep:
                                   isInitialStep = False
                     
                     ti += 1
                     thisTime += thisDt
                     diagTime += thisDt
              
       #%%       
       endt = time.time()
       print('Solve the system: DONE!')
       print('Elapsed time: ', endt - start)
              
       #%% Recover the solution (or check the residual)
       GS = ZTL.shape
       uxz = np.reshape(fields[:,0], GS, order='F') 
       wxz = np.reshape(fields[:,1], GS, order='F')
       pxz = np.reshape(fields[:,2], GS, order='F') 
       txz = np.reshape(fields[:,3], GS, order='F')
       
       #%% Make some plots for static or transient solutions
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
              
              flowAngle = np.arctan(wxz[0,:] * np.reciprocal(hydroState[ebcDex[2],0] + uxz[0,:]))
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
                     dqdt = np.reshape(rhsVec[:,pp], (NZ, NX), order='F')
                     ccheck = plt.contourf(1.0E-3*XL, 1.0E-3*ZTL, dqdt, 201, cmap=cm.seismic)
                     plt.colorbar(ccheck, format='%+.3E')
                     plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                     plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
                     plt.tight_layout()
              plt.show()
              
              # Check W at the boundaries...
              return (XL, ZTL, rhsVec)
       
if __name__ == '__main__':
       
       #TestName = 'ClassicalSchar01'
       #TestName = 'ClassicalScharIter'
       #TestName = 'UniformStratStatic'
       #TestName = 'DiscreteStratStatic'
       TestName = 'UniformTestTransient'
       #TestName = '3LayerTestTransient'
       
       # Run the model in a loop if needed...
       for ii in range(1):
              diagOutput = runModel(TestName)
