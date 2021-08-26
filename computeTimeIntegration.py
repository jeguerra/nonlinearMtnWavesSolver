#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
import scipy.sparse as sps
import time as timing
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

def enforceEssentialBC(sol, init, zeroDex, ebcDex, DZDX):
       
       # Enforce essential boundary conditions
       sol[zeroDex[0],0] = np.zeros(len(zeroDex[0]))
       sol[zeroDex[1],1] = np.zeros(len(zeroDex[1]))
       sol[zeroDex[2],2] = np.zeros(len(zeroDex[2]))
       sol[zeroDex[3],3] = np.zeros(len(zeroDex[3]))
       
       #bdex = ebcDex[1]
       #U = sol[:,0] + init[:,0]
       #sol[bdex,1] = np.array(DZDX[bdex,0] * U[bdex])
       
       return sol

def computeTimeIntegrationNL2(DIMS, PHYS, REFS, REFG, DLD, DLD2, TOPT, \
                              sol0, dsol0, init0, zeroDex, ebcDex, \
                              DynSGS, DCF, isFirstStep):
       
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       
       DT = TOPT[0]
       order = TOPT[3]
       RdT_bar = REFS[9][0]
           
       mu = REFG[3]
       RLM = REFG[4]
       DZDX = REFS[15]
       
       diffusiveFlux = False
       #'''
       if isFirstStep:
              # Use SciPY sparse for dynamics
              DDXM_A = REFS[10][0]
              DDZM_A = REFS[10][1]
       else:
              # Use multithreading on CPU and GPU
              DDXM_A = REFS[12][0]
              DDZM_A = REFS[12][1]
       
       DDXM_B = REFS[13][0]
       DDZM_B = REFS[13][1]
                     
       def computeUpdate(coeff, solA, sol2Update):
              tol = 1.0E-15
              DF = coeff * DT
              
              RayDamp = np.reciprocal(1.0 + (mu * DF) * RLM.data)
              
              # Append log perturbation u and w... for advection
              U = solA[:,0] + init0[:,0]
              W = solA[:,1]
              
              # Compute first derivatives
              DqDx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_A, DDZM_A)
              
              # Numerical "clean up" here
              DqDx[np.abs(DqDx) < tol] = 0.0
              DqDz[np.abs(DqDz) < tol] = 0.0
              
              # Compute advective update (explicit)
              #rhsDyn = computeRHSUpdate_dynamics(solA, U, W, DqDx, DqDz, coeff * DT)
              args1 = [PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, solA, U, W, ebcDex, zeroDex]
              rhsAdv = tendency.computeEulerEquationsLogPLogT_Advection(*args1)
              
              # Apply explicit part of the update
              solAdv = solA + DF * rhsAdv
              
              # Compute internal forces (semi implicit)
              args2 = [PHYS, rhsAdv, DqDx, DqDz, REFG, DZDX, RdT_bar, solAdv, ebcDex, zeroDex]
              rhsFrc = tendency.computeEulerEquationsLogPLogT_InternalForce(*args2)
              
              solB = sol2Update + DF * (rhsAdv + rhsFrc)
              #'''
              # Compute 2nd derivatives
              if diffusiveFlux:
                     P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz = \
                     tendency.computeFieldDerivativesFlux(DqDx, DqDz, DCF, REFG, DDXM_B, DDZM_B, DZDX)
              else:
                     P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz = \
                     tendency.computeFieldDerivatives2(DqDx, DqDz, REFG, DDXM_B, DDZM_B, DZDX)
                     
              # Compute diffusive update (explicit)
              rhsDif = computeRHSUpdate_diffusion(solA, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz)
              
              # Apply diffusion update
              solB += DF * rhsDif
              
              # Apply Rayleigh layer implicitly
              solB = (RayDamp.T) * solB
              #'''
              #solB = enforceEssentialBC(solB, init0, zeroDex, ebcDex, DZDX)
              
              # Filter the solution to prevent underflow
              solB[np.abs(solB) < tol] = 0.0
              
              return solB
       
       def computeRHSUpdate_dynamics(sol, U, W, DqDx, DqDz, DT):
              
              # Compute dynamical tendencies
              #rhs = tendency.computeEulerEquationsLogPLogT_Explicit(PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, sol, U, W, ebcDex, zeroDex)
              #'''
              args1 = [PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, sol, U, W, ebcDex, zeroDex]
              rhsAdvection = tendency.computeEulerEquationsLogPLogT_Advection(*args1)
              
              solAdv = sol + DT * rhsAdvection
              #DqDx, DqDz = tendency.computeFieldDerivatives(solAdv, DDXM_A, DDZM_A)
              
              args2 = [PHYS, rhsAdvection, DqDx, DqDz, REFG, DZDX, RdT_bar, solAdv, ebcDex, zeroDex]
              rhsInternalF = tendency.computeEulerEquationsLogPLogT_InternalForce(*args2)
              
              rhs = rhsAdvection + rhsInternalF
              #'''
              return rhs
       
       def computeRHSUpdate_diffusion(fields, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz):
              
              if diffusiveFlux:
                     rhs = tendency.computeDiffusiveFluxTendency(PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                                      REFS, ebcDex, zeroDex, DynSGS)
              else:
                     rhs = tendency.computeDiffusionTendency(PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                                      REFS, ebcDex, zeroDex, DCF, DynSGS)
       
              return rhs
       
       def ssprk43(sol):
              # Stage 1
              sol1 = computeUpdate(0.5, sol, sol)
              # Stage 2
              sol2 = computeUpdate(0.5, sol1, sol1)
              # Stage 3
              sol = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              sol1 = computeUpdate(1.0 / 6.0, sol, sol)
              # Stage 4
              sol = computeUpdate(0.5, sol, sol)
              
              return sol
       
       def ssprk53_Opt(sol):
              # Optimized truncation error to SSP coefficient method from Higueras, 2019
              # Stage 1
              sol1 = computeUpdate(0.377268915331368, sol, sol)
              # Stage 2
              sol2 = computeUpdate(0.377268915331368, sol1, sol1)
              # Stage 3
              sol3 = np.array(0.426988976571684 * sol + 0.5730110234283154 * sol2)
              sol2 = computeUpdate(0.216179247281718, sol2, sol3)
              # Stage 4
              sol3 = np.array(0.193245318771018 * sol + 0.199385926238509 * sol1 + 0.607368754990473 * sol2)
              sol2 = computeUpdate(0.229141351401419, sol2, sol3)
              # Stage 5
              sol3 = np.array(0.108173740702208 * sol1 + 0.891826259297792 * sol2)
              sol = computeUpdate(0.336458325509300, sol2, sol3)
              
              return sol
       
       def RK64_NL(sol):
              # Stage 1
              omega = computeUpdate(0.032918605146, sol, 0.0)
              sol += omega
              # Stage 2
              omega = computeUpdate(1.0, sol, -0.737101392796 * omega)
              sol += 0.8232569982 * omega
              # Stage 3
              omega = computeUpdate(1.0, sol, -1.634740794341 * omega)
              sol += 0.3815309489 * omega
              # Stage 4
              omega = computeUpdate(1.0, sol, -0.744739003780 * omega)
              sol += 0.200092213184 * omega
              # Stage 5
              omega = computeUpdate(1.0, sol, -1.469897351522 * omega)
              sol += 1.718581042715 * omega
              # Stage 6
              omega = computeUpdate(1.0, sol, -2.813971388035 * omega)
              sol += 0.27 * omega
              # Stage 7
              sol = computeUpdate(1.0 - 0.847252983783, sol, sol)
              
              # third output is a factor on DT: T_new = T_old + 0.85 * DT
              # this integrator does not move a whole time step...
              return sol
       
       def ketcheson62(sol):
              m = 5
              c1 = 1 / (m-1)
              c2 = 1 / m
              sol1 = np.array(sol)
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
              
              sol = computeUpdate(c1, sol, sol, False)
              sol1 = np.array(sol)
                     
              for ii in range(4):
                     sol = computeUpdate(c1, sol, sol, False)
                     
              # Compute stage 6 with linear combination
              sol1 = np.array(0.6 * sol1 + 0.4 * sol)
              sol = computeUpdate(c2, sol, sol1, False)
              
              for ii in range(3):
                     sol= computeUpdate(c1, sol, sol, False)
                     
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
       
       return (solB - sol0)