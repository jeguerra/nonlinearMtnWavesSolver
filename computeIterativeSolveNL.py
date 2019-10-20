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
       linSol = SOLT
       
       def computeRHSUpdate(sol):
              fields, uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, botdex, topdex)
              rhs += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       
              # Multiply by -1 here. The RHS was computed for transient solution
              return -1.0 * rhs
       
       # Solve for nonlinear equilibrium
       sol = root(computeRHSUpdate, linSol, method='krylov', \
                  options={'disp':True, 'maxiter':1000, 'jac_options':{'inner_maxiter':100,'method':'lgmres','outer_k':100}})
       '''
       for pp in range(10):
              sol = root(computeRHSUpdate, sol.x, method='df-sane', \
                         options={'disp':True, 'maxfev':500, 'M':10, 'line_search':'cheng'})
              
              sol = root(computeRHSUpdate, sol.x, method='krylov', \
                         options={'disp':True, 'maxiter':20, 'jac_options':{'inner_maxiter':50,'method':'lgmres','outer_k':100}})
       '''
       print('NL solver exit on: ', sol.message)
       print('Number of NL solver iterations: ', sol.nit)
       
       return sol.x