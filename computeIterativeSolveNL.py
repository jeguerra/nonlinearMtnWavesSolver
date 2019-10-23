#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:05:20 2019

@author: jorge.guerra
"""
import time
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.optimize import root
import scipy.sparse.linalg as spl
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

def computeIterativeSolveNL(PHYS, REFS, REFG, DX, DZ, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex, sysDex, DynSGS):
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
       sol0 = linSol
       #sol0 = root(computeRHSUpdate, linSol, method='krylov', \
       #           options={'disp':True, 'maxiter':5, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
       F0 = computeRHSUpdate(sol0)
       sol1 = root(computeRHSUpdate, sol0, method='krylov', \
                  options={'disp':True, 'maxiter':5, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
       F1 = computeRHSUpdate(sol1.x)
       # Compute the differences
       DF = F1 - F0
       DSOL = sol1.x - sol0
       del(F0); del(sol0)
       del(F1); del(sol1)
       #plt.plot(DSOL)
       #plt.show()
       #plt.plot(DF)
       #plt.show()
       # Apply the BC indeces
       #DF = DF[sysDex]
       #DSOL = DSOL[sysDex]
       # Compute the Jacobian matrix by columns
       M = len(DF)
       JAC = sps.lil_matrix((M,M))
       # Exclude places where there is little change in the solution
       dtol_fact = 1.0E-2 # 1% of maximum absolute value change in EACH variable
       drop_tol = dtol_fact * np.amax(np.abs(DSOL[udex])) # small horizontal wind change
       nzDex = np.where(np.abs(DSOL[udex]) > drop_tol)
       unzDex = udex[nzDex]
       drop_tol = dtol_fact * np.amax(np.abs(DSOL[wdex])) # small vertical wind change
       nzDex = np.where(np.abs(DSOL[wdex]) > drop_tol)
       wnzDex = wdex[nzDex]
       drop_tol = dtol_fact * np.amax(np.abs(DSOL[pdex])) # small log pressure change
       nzDex = np.where(np.abs(DSOL[pdex]) > drop_tol)
       pnzDex = pdex[nzDex]
       drop_tol = dtol_fact * np.amax(np.abs(DSOL[tdex])) # small log potential temp. change
       nzDex = np.where(np.abs(DSOL[tdex]) > drop_tol)
       tnzDex = tdex[nzDex]
       
       nzDex = np.concatenate((unzDex, wnzDex, pnzDex, tnzDex))
       IDSOL = np.reciprocal(DSOL[nzDex])
       NZS = len(nzDex)
       print(M, NZS)
       
       for ii in range(M):
              #print(M, len(nzDex))
              jac_row = (DF[ii] / NZS) * IDSOL
              # Compute maximum change for this row and apply drop tolerance
              #mc = np.amax(np.abs(jac_row))
              #jac_row[np.abs(jac_row) < drop_tol * mc] = 0.0
              # Set the nonzero elements of this row
              JAC[ii,nzDex] = jac_row #np.expand_dims(jac_row, axis=1)
                     
       JAC = JAC.tocsc()
       plt.spy(JAC)
       plt.show()
       
       end = time.time()
       print('Compute numerical approximation of local Jacobian... DONE!')
       print('Time to compute local Jacobian matrix: ', end - start)
       start = time.time()
       #'''
       opts = dict(Equil=True, IterRefine='DOUBLE')
       factor = spl.splu(JAC, permc_spec='MMD_ATA', options=opts)
       end = time.time()
       #'''    
       end = time.time()          
       print('Compute numerical approximation of local Jacobian... DONE!')
       print('Time to compute local Jacobian factorization: ', end - start)
       
       #'''
       # Solve for nonlinear equilibrium
       #sol = root(computeRHSUpdate, SOLT[:,0], method='hybr', jac=False, tol=1.0E-6)
       sol = root(computeRHSUpdate, SOLT[:,0], method='krylov', \
                  options={'disp':True, 'maxiter':1000, 'jac_options':{'inner_maxiter':20,'method':'lgmres','outer_k':10}})
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