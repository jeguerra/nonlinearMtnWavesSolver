#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:05:20 2019

@author: jorge.guerra
"""

import numpy as np
from scipy.optimize import root
import computeEulerEquationsLogPLogT as tendency

def computePrepareFields(PHYS, REFS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex):
       # Get some physical quantities
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       
       # Get the boundary terrain
       #dHdX = REFS[6]
       
       # Get the solution components
       uxz = SOLT[udex]
       wxz = SOLT[wdex]
       pxz = SOLT[pdex]
       txz = SOLT[tdex]
       
       # Make the total quatities
       U = uxz + INIT[udex]
       LP = pxz + INIT[pdex]
       LT = txz + INIT[tdex]
       
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       fields = np.empty((len(uxz), 4))
       fields[:,0] = uxz 
       fields[:,1] = wxz
       fields[:,2] = pxz
       fields[:,3] = txz
       
       return fields, uxz, wxz, pxz, txz, U, RdT

def computeIterativeSolveNL(PHYS, REFS, REFG, DX, DZ, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex, DynSGS):
       linSol = SOLT[:,0]
       
       def computeRHSUpdate(sol):
              fields, uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, botdex, topdex)
              rhs += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       
              # Multiply by -1 here. The RHS was computed for transient solution
              return -1.0 * rhs
       
       # Approximate the Jacobian numerically...
       SOLT[:,0] = root(computeRHSUpdate, linSol, method='krylov', jac=False, \
                  options={'disp':False, 'maxiter':10, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
       F0 = computeRHSUpdate(SOLT[:,0])
       SOLT[:,1] = root(computeRHSUpdate, SOLT[:,0], method='krylov', jac=False, \
                  options={'disp':False, 'maxiter':10, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
       F1 = computeRHSUpdate(SOLT[:,1])
       # Compute the differences
       DF = F1 - F0
       DSOL = SOLT[:,1] - SOLT[:,0]
       del(F0)
       del(F1)
       # Compute the Jacobian matrix
       M = len(DF)
       JAC = np.empty((M,M))
       for ii in range(M):
              for jj in range(M):
                     if abs(DSOL[jj]) > 0.0:
                            JAC[ii,jj] = DF[ii] / DSOL[jj]
       
       # Solve for nonlinear equilibrium
       sol = root(computeRHSUpdate, SOLT[:,0], method='hybr', jac=False, tol=1.0E-6)
       #sol = root(computeRHSUpdate, SOLT[:,0], method='krylov', jac=False, \
       #           options={'disp':True, 'maxiter':100, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
       '''
       for pp in range(10):
              sol = root(computeRHSUpdate, sol.x, method='df-sane', \
                         options={'disp':True, 'maxfev':100, 'M':10, 'line_search':'cruz'})
              
              sol = root(computeRHSUpdate, sol.x, method='krylov', \
                         options={'disp':True, 'maxiter':20, 'jac_options':{'inner_maxiter':50,'method':'lgmres','outer_k':10}})
       '''
       print('NL solver exit on: ', sol.message)
       print('Number of NL solver iterations: ', sol.nit)
       
       return sol.x