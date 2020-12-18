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
import computeResidualViscCoeffs as rescf
import computeEulerEquationsLogPLogT as tendency

def rampFactor(time, timeBound):
       if time == 0.0:
              uRamp = 0.0
              DuRamp = 0.0
       elif time <= timeBound:
              uRamp = 0.5 * (1.0 - mt.cos(mt.pi / timeBound * time))
              DuRamp = 0.5 * mt.pi / timeBound * mt.sin(mt.pi / timeBound * time)
              #uRamp = mt.sin(0.5 * mt.pi / timeBound * time)
              #uRamp = uRamp**4
       else:
              uRamp = 1.0
              DuRamp = 0.0
              
       return uRamp, DuRamp

def plotRHS(time, rhs, ebcDex):
       plt.figure(figsize=(10.0, 10.0))
       plt.subplot(2,2,1)
       plt.plot(rhs[ebcDex[1],0]); plt.plot(rhs[ebcDex[1]+1,0])
       plt.xlim(150, 300); plt.legend(('dudt at BC', 'dudt at level 1'))
       plt.subplot(2,2,2)
       plt.plot(rhs[ebcDex[1],1]); plt.plot(rhs[ebcDex[1]+1,1])
       plt.xlim(150, 300); plt.legend(('dwdt at BC', 'dwdt at level 1'))
       plt.subplot(2,2,3)
       plt.plot(rhs[ebcDex[1],2]); plt.plot(rhs[ebcDex[1]+1,2])
       plt.xlim(150, 300); plt.legend(('dpdt at BC', 'dpdt at level 1'))
       plt.subplot(2,2,4)
       plt.plot(rhs[ebcDex[1],3]); plt.plot(rhs[ebcDex[1]+1,3])
       plt.xlim(150, 300); plt.legend(('dtdt at BC', 'dtdt at level 1'))
       plt.show()
       
       return

def computeTimeIntegrationNL2(PHYS, REFS, REFG, DLS, DXD, DZD, TOPT, \
                              sol0, init0, rhs0, zeroDex, ebcDex, \
                              DynSGS, DCF, thisTime, isFirstStep):
       
       DX2 = DXD * DXD
       DZ2 = DZD * DZD
       DXZ = DXD * DZD
       DT = TOPT[0]
       rampTimeBound = TOPT[2]
       order = TOPT[3]
       RdT_bar = REFS[9][0]
       P_bar = REFS[9][1]
       
       # Adjust for time ramping
       time = thisTime
       uf, duf = rampFactor(time, rampTimeBound)
           
       mu = REFG[3]
       RLM = REFG[4]
       DQDZ = REFG[2]
       #DQDZ = GMLZ.dot(REFG[2])
       DZDX = REFS[15]
       #'''
       if isFirstStep:
              # Use SciPY sparse for dynamics
              DDXM_CPU = REFS[10]
              DDZM_CPU = REFS[11]
              DDXM_CFD = REFS[13][0]
              DDZM_CFD = REFS[13][1]
       else:
              # Use multithreading on CPU and GPU
              DDXM_CPU = REFS[12][0]
              DDZM_CPU = REFS[12][1]
              DDXM_CFD = REFS[13][0]
              DDZM_CFD = REFS[13][1]
       
       def computeUpdate(coeff, solA, sol2Update):
              
              # Compute 1st derivatives
              DqDx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_CPU, DDZM_CPU)
              
              # Compute 2nd derivatives
              D2qDx2, D2qDz2, D2qDxz, PqPx, PqPz = \
                     tendency.computeFieldDerivatives2(DqDx, DqDz, DQDZ, DDXM_CFD, DDZM_CFD, DZDX)
              
              # Compute dynamics update
              rhsDyn = computeRHSUpdate_dynamics(solA, DqDx, DqDz)
              
              # Compute the diffusion update
              rhsDif = computeRHSUpdate_diffusion(PqPx, PqPz, D2qDx2, D2qDz2, D2qDxz, DCF)
              
              # Apply update
              dsol = coeff * DT * (rhsDyn + rhsDif)
              solB = sol2Update + dsol
              
              #''' TURNED ON IN ORIGINAL RUN
              # Apply Rayleigh layer implicitly
              propagator = np.reciprocal(1.0 + (mu * coeff * DT) * RLM.data)
              solB = propagator.T * solB
              #'''
              
              return solB, rhsDyn
       
       def computeRHSUpdate_dynamics(fields, DqDx, DqDz):
              U = fields[:,0] + init0[:,0]
              W = fields[:,1] + init0[:,1]
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DZDX, RdT_bar, fields, U, W, ebcDex, zeroDex)
                     
              return rhs
       
       def computeRHSUpdate_diffusion(DqDx, DqDz, D2qDx2, D2qDz2, D2qDxz, dcoeff):
              
              rhs = tendency.computeDiffusionTendency(PHYS, dcoeff, DqDx, DqDz, D2qDx2, D2qDz2, D2qDxz, DZDX, ebcDex, DX2, DZ2, DXZ)
       
              return rhs
       
       def computeDCFUpdate(solA, solB, rhsA, rhsB):
              # Compute sound speed
              T_ratio = np.exp(PHYS[4] * solB[:,2] + solB[:,3]) - 1.0
              RdT = REFS[9][0] * (1.0 + T_ratio)
              #PZ = np.exp(solB[:,2] + init0[:,2])
              #RHOI = RdT * np.reciprocal(PZ)
              VSND = np.sqrt(PHYS[6] * RdT)
              # Compute flow speed
              UD = solB[:,0] + init0[:,0]
              WD = solB[:,1]
              vel = np.stack((UD, WD),axis=1)
              VFLW = np.linalg.norm(vel, axis=1)
              # Compute total wave speed
              VWAV = VFLW + VSND
              # Compute max norm of total wave speed
              VWAV_max = bn.nanmax(VWAV)
              DTN = DLS / VWAV_max
              
              # Trapezoidal Rule estimate of residual
              dqdt = (1.0 / TOPT[0]) * (solB - solA)
              resInv = dqdt - 0.5 * (rhsA + rhsB)
              
              # Compute DynSGS or Flow Dependent diffusion coefficients
              QM = bn.nanmax(np.abs(solf - bn.nanmean(solf)), axis=0)
              DCF = rescf.computeResidualViscCoeffs(resInv, QM, VFLW, DXD, DZD, DX2, DZ2, 1.0)                     
              
              return DCF, resInv, rhsB, DTN
       
       def ssprk34(sol):
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
              sol1, rhs = computeUpdate(0.377268915331368, sol, sol)
              # Stage 2
              sol2, rhs = computeUpdate(0.377268915331368, sol1, sol1)
              # Stage 3
              sol3 = np.array(0.426988976571684 * sol + 0.5730110234283154 * sol2)
              sol2, rhs = computeUpdate(0.216179247281718, sol2, sol3)
              # Stage 4
              sol3 = np.array(0.193245318771018 * sol + 0.199385926238509 * sol1 + 0.607368754990473 * sol2)
              sol2, rhs = computeUpdate(0.229141351401419, sol2, sol3)
              # Stage 5
              sol3 = np.array(0.108173740702208 * sol1 + 0.891826259297792 * sol2)
              sol, rhs = computeUpdate(0.336458325509300, sol2, sol3)
              
              return sol, rhs
       
       def RK64_NL(sol):
              # Stage 1
              omega, rhs = computeUpdate(0.032918605146, sol, 0.0)
              sol += omega
              # Stage 2
              omega, rhs = computeUpdate(1.0, sol, -0.737101392796 * omega)
              sol += 0.8232569982 * omega
              # Stage 3
              omega, rhs = computeUpdate(1.0, sol, -1.634740794341 * omega)
              sol += 0.3815309489 * omega
              # Stage 4
              omega, rhs = computeUpdate(1.0, sol, -0.744739003780 * omega)
              sol += 0.200092213184 * omega
              # Stage 5
              omega, rhs = computeUpdate(1.0, sol, -1.469897351522 * omega)
              sol += 1.718581042715 * omega
              # Stage 6
              omega, rhs = computeUpdate(1.0, sol, -2.813971388035 * omega)
              sol += 0,27 * omega
              # Stage 7
              sol, rhs = computeUpdate(1.0 - 0.847252983783, sol, sol)
              
              # third output is a factor on DT: T_new = T_old + 0.85 * DT
              # this integrator does not move a whole time step...
              return sol, rhs
       
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
              
              sol, rhs = computeUpdate(c1, sol, sol)
              sol1 = np.array(sol)
                     
              for ii in range(4):
                     sol, rhs = computeUpdate(c1, sol, sol)
                     
              # Compute stage 6 with linear combination
              sol1 = np.array(0.6 * sol1 + 0.4 * sol)
              sol, rhs = computeUpdate(c2, sol, sol1)
              
              for ii in range(3):
                     sol, rhs = computeUpdate(c1, sol, sol)
                     
              return sol, rhs
       
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
              solf, rhsf = ketcheson62(sol0)
       elif order == 3:
              solf, rhsf = ketcheson93(sol0)
              #solf = kinnGray53(sol0)
              #solf, rhsf = ssprk53_Opt(sol0)
       elif order == 4:
              solf = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solf, rhsDyn = ketcheson62(sol0)
       
       time += DT
       DCF, resInv, rhsf, DTN = computeDCFUpdate(sol0, solf, rhs0, rhsf)
              
       #input('End of time step. ' + str(time))
       
       return solf, rhsf, time, resInv, DCF, DTN