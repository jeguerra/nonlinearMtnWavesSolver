#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:05:20 2019

@author: jorge.guerra
"""
import time
import numpy as np
#import scipy.linalg as dsl
#import scipy.sparse as sps
#import matplotlib.pyplot as plt
import scipy.optimize as opt
#import scipy.sparse.linalg as spl
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

def computeIterativeSolveNL(PHYS, REFS, REFG, DX, DZ, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex, sysDex, INV_LDU):
       linSol = SOLT[:,0]
       
       def computeRHSUpdate(sol):
              fields, uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, uxz, wxz, pxz, txz, U, RdT, botdex, topdex)
              rhs += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       
              # Multiply by -1 here. The RHS was computed for transient solution
              return -1.0 * rhs
       #'''
       # Approximate the Jacobian numerically...
       start = time.time()
       F_lin = computeRHSUpdate(linSol)
       # Perturb the solution by iterative method
       sol0 = opt.root(computeRHSUpdate, linSol, method='krylov', \
                  options={'disp':True, 'maxiter':5, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
       F0 = computeRHSUpdate(sol0.x)
       # Compute the differences
       DF = F0 - F_lin
       DSOL = sol0.x - linSol
       DF = DF[sysDex]
       DSOL = DSOL[sysDex]
       #del(F0); del(sol0)
       #del(F1); del(sol1)
       # Index only places where change is happening
       nzDex = np.where(DSOL > 0.0)
       nzDex = nzDex[0]
       # Get inverse Jacobian at these DOF only
       IJAC_L = (INV_LDU[0].toarray())[np.ix_(nzDex, nzDex)]
       IJAC_D = (INV_LDU[1].toarray())[np.ix_(nzDex, nzDex)]
       IJAC_U = (INV_LDU[2].toarray())[np.ix_(nzDex, nzDex)]
       IJAC = IJAC_D.dot(IJAC_U)
       IJAC = IJAC_L.dot(IJAC)
       
       # Compute update  to inverse Jacobian (Sherman-Morrison)
       num = DSOL[nzDex] - IJAC.dot(DF[nzDex])
       dsol = np.expand_dims(DSOL[nzDex], axis=0)
       chg = dsol.dot(IJAC)
       df = np.expand_dims(DSOL[nzDex], axis=1)
       scale = chg.dot(df)
       DJ = (1.0 / scale[0,0]) * np.outer(num, chg.flatten())
       
       # Compute update to inverse Jacobian
       IJAC += DJ
       
       solNewton = sol0.x
       for nn in range(5):
              dsol = -IJAC.dot(F0[nzDex])
              solNewton[nzDex] += dsol
              F0 = computeRHSUpdate(solNewton)
              print(np.linalg.norm(F0))
       
       #'''
       # Solve for nonlinear equilibrium
       #sol = root(computeRHSUpdate, sol1.x, method='broyden1', jac=False, options={'disp':True})
       sol = opt.root(computeRHSUpdate, linSol, method='krylov', \
                  options={'disp':True, 'maxiter':5000, 'jac_options':{'inner_maxiter':50,'method':'lgmres','outer_k':10}})
       
       '''
       for pp in range(10):
              sol = root(computeRHSUpdate, sol.x, method='df-sane', \
                         options={'disp':True, 'maxfev':50, 'M':10, 'line_search':'cruz'})
              
              sol = root(computeRHSUpdate, sol.x, method='krylov', \
                         options={'disp':True, 'maxiter':20, 'jac_options':{'inner_maxiter':50,'method':'lgmres','outer_k':10}})
       '''
       print('NL solver exit on: ', sol.message)
       print('Number of NL solver iterations: ', sol.nit)
       
       return sol.x