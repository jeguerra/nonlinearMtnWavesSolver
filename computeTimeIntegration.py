#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
import scipy.sparse as sps
import time as tm
import bottleneck as bn
import matplotlib.pyplot as plt
import computeEulerEquationsLogPLogT as tendency
import computeResidualViscCoeffs as rescf

def plotRHS(x, rhs, ebcDex, label):
       plim = 2.0E4
       plt.figure(figsize=(10.0, 10.0))
       plt.title(label)
       plt.subplot(2,2,1)
       plt.plot(x, rhs[ebcDex[1],0]); plt.plot(x, rhs[ebcDex[1]+1,0])
       plt.xlim(-plim, plim)
       plt.legend(('level 0', 'level 1'))
       plt.subplot(2,2,2)
       plt.plot(x, rhs[ebcDex[1],1]); plt.plot(x, rhs[ebcDex[1]+1,1])
       plt.xlim(-plim, plim) 
       plt.legend(('level 0', 'level 1'))
       plt.subplot(2,2,3)
       plt.plot(x, rhs[ebcDex[1],2]); plt.plot(x, rhs[ebcDex[1]+1,2])
       plt.xlim(-plim, plim) 
       plt.legend(('level 0', 'level 1'))
       plt.subplot(2,2,4)
       plt.plot(x, rhs[ebcDex[1],3]); plt.plot(x, rhs[ebcDex[1]+1,3])
       plt.xlim(-plim, plim) 
       plt.legend(('level 0', 'level 1'))
       plt.show()
       
       return

def enforceEssentialBC(sol, init, zeroDex, ebcDex, dhdx):
              
       # Enforce essential boundary conditions
       sol[zeroDex[0],0] = 0.0
       sol[zeroDex[1],1] = 0.0
       sol[zeroDex[2],2] = 0.0
       sol[zeroDex[3],3] = 0.0
       
       bdex = ebcDex[2]
       U = sol[:,0] + init[:,0]
       sol[bdex,1] = np.array(dhdx * U[bdex])
       
       return sol

def computeTimeIntegrationNL2(DIMS, PHYS, REFS, REFG, DLD, TOPT, \
                              sol0, init0, zeroDex, ebcDex, \
                              isFirstStep, filteredCoeffs, NE):
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       RLM = REFG[4]
       
       diffusiveFlux = False
       
       #'''
       if isFirstStep:
              # Use SciPY sparse for dynamics
              DDXM_A = REFS[10][0]
              #DDZM_A = REFS[10][1]
              #DDXM_A = REFS[17]
              DDZM_A = REFS[19]
       else:
              # Use multithreading on CPU
              DDXM_A = REFS[12][0]
              #DDZM_A = REFS[12][1]
              #DDXM_A = REFS[18]
              DDZM_A = REFS[20]
       
       DDXM_B = REFS[13][0]
       DDZM_B = REFS[13][1]
                     
       def computeUpdate(coeff, solA, sol2Update):
              DF = coeff * DT
              
              # Append log perturbation u and w... for advection
              U = solA[:,0] + init0[:,0]
              W = solA[:,1]
              
              # Compute first derivatives and RHS with spectral operators
              #DqDxA, DqDzA = tendency.computeFieldDerivatives(solA, DDXM_A, DDZM_A)
              DqDxA, DqDzA = tendency.computeFieldDerivativeStag(solA, DDXM_A, DDZM_A)
              
              '''
              args1 = [PHYS, DqDxA, DqDzA, REFS, REFG, solA, U, W]
              rhsAdv = tendency.computeEulerEquationsLogPLogT_Explicit(*args1)
              rhsAdv = tendency.enforceTendencyBC(rhsAdv, zeroDex, ebcDex, REFS[6][0])
              '''
              #'''
              # Compute advective update (explicit)
              args1 = [PHYS, DqDxA, DqDzA, REFS, REFG, solA, U, W]
              rhsAdv = tendency.computeEulerEquationsLogPLogT_Advection(*args1)
              rhsAdv = tendency.enforceTendencyBC(rhsAdv, zeroDex, ebcDex, REFS[6][0])
              
              # Apply explicit part of the update
              solAdv = sol2Update + DF * rhsAdv
              solAdv = enforceEssentialBC(solAdv, init0, zeroDex, ebcDex, REFS[6][0])
              # Compute internal forces (semi implicit)
              
              args2 = [PHYS, DqDxA, DqDzA, REFS, REFG, solAdv]
              rhsFrc = tendency.computeEulerEquationsLogPLogT_InternalForce(*args2)
              rhsFrc = tendency.enforceTendencyBC(rhsFrc, zeroDex, ebcDex, REFS[6][0])
              
              solB = solAdv + DF * rhsFrc
              solB = enforceEssentialBC(solB, init0, zeroDex, ebcDex, REFS[6][0])
              #'''
              
              # Update the adaptive coefficients
              #resField = np.copy(rhsAdv + rhsFrc)
              DqDxR, DqDzR = tendency.computeFieldDerivatives(solB, DDXM_B, DDZM_B)
              args1 = [PHYS, DqDxR, DqDzR, REFS, REFG, solB, U, W]
              resField = tendency.computeEulerEquationsLogPLogT_Explicit(*args1)
              if filteredCoeffs:
                     DCF = rescf.computeResidualViscCoeffsFiltered(DIMS, resField, solB, init0, DLD, NE)
              else:
                     DCF = rescf.computeResidualViscCoeffsRaw(DIMS, resField, solB, init0, DLD, REFS[6][0], ebcDex[2])
              del(resField)
              
              # Compute diffusive tendency
              P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz = \
              tendency.computeFieldDerivatives2(DqDxA, DqDzA, DDXM_B, DDZM_B, REFS, REFG, DCF, diffusiveFlux)
              
              rhsDif = tendency.computeDiffusionTendency(solA, DqDxA, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, zeroDex, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply diffusion update
              #solB = sol2Update + DF * (rhsAdv + rhsDif)
              solB += DF * rhsDif
              solB = enforceEssentialBC(solB, init0, zeroDex, ebcDex, REFS[6][0])
              
              # Apply Rayleigh damping layer implicitly
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM.data)
              #npdex = np.array([0,1,2,3])
              solB = np.copy(RayDamp.T * solB)
                            
              return solB
       
       def ssprk43(sol):
              # Stage 1
              sol1 = computeUpdate(0.5, sol, sol)
              # Stage 2
              sol2 = computeUpdate(0.5, sol1, sol1)
              # Stage 3
              sol = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              sol1 = computeUpdate(1.0 / 6.0, sol, sol)
              # Stage 4
              sol = computeUpdate(0.5, sol1, sol1)
              
              return sol
       
       def ssprk53_Opt(sol):
              # Optimized truncation error to SSP coefficient method from Higueras, 2019
              # Stage 1
              sol1 = computeUpdate(0.377268915331368, sol, sol)
              # Stage 2
              sol2 = computeUpdate(0.377268915331368, sol1, sol1)
              # Stage 3
              sol3 = np.array(0.568582304164742 * sol + 0.431417695835258 * sol2)
              sol3 = computeUpdate(0.162760486162526, sol2, sol3)
              # Stage 4
              sol4 = np.array(0.088796463619276 * sol + 0.000050407140024 * sol1 + 0.9111531292407 * sol3)
              sol4 = computeUpdate(0.343749752769421, sol3, sol4)
              # Stage 5
              sol5 = np.array(0.210401429751688 * sol1 + 0.789598570248313 * sol4)
              sol = computeUpdate(0.29789099614478, sol4, sol5)
              
              return sol
       
       def ketcheson62(sol):
              m = 5
              c1 = 1 / (m-1)
              c2 = 1 / m
              sol1 = np.copy(sol)
              for ii in range(m):
                     if ii == m-1:
                            sol1 = c2 * ((m-1) * sol + sol1)
                            sol = computeUpdate(c2, sol, sol1)
                     else:
                            sol = computeUpdate(c1, sol, sol)
                      
              return sol
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 15.0
              
              sol = computeUpdate(c1, sol, sol)
              sol1 = np.copy(sol)
              
              for ii in range(4):
                     sol = computeUpdate(c1, sol, sol)
                     
              # Compute stage 6 with linear combination
              sol1 = np.array(0.6 * sol1 + 0.4 * sol)
              sol = computeUpdate(c2, sol, sol1)
              
              for ii in range(3):
                     sol= computeUpdate(c1, sol, sol)
                     
              return sol
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
       
              sol2 = np.array(sol)
              for ii in range(5):
                     sol = computeUpdate(c1, sol, sol)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     sol = computeUpdate(c1, sol, sol)
                     
              sol2 = sol2 + 0.6 * sol
              sol = computeUpdate(0.1, sol, sol2)
              
              return sol

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketcheson62(sol0)
       elif order == 3:
              #solB = ketcheson93(sol0)
              #solB = ssprk43(sol0)
              solB = ssprk53_Opt(sol0)
       elif order == 4:
              solB = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketcheson62(sol0)
       
       return solB