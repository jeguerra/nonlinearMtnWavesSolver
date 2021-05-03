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

def computeTimeIntegrationNL2(DIMS, PHYS, REFS, REFG, DX2, DZ2, DXZ, TOPT, \
                              sol0, init0, zeroDex, ebcDex, \
                              DynSGS, DCF, thisTime, isFirstStep):
       
       DT = TOPT[0]
       rampTimeBound = TOPT[2]
       order = TOPT[3]
       RdT_bar = REFS[9][0]
       
       # Adjust for time ramping
       uf, duf = rampFactor(thisTime, rampTimeBound)
       
       # Get the GML factors
       #GMLX = REFG[0][1]
       #GMLZ = REFG[0][2]
           
       mu = REFG[3]
       RLM = REFG[4]
       #DQDZ = REFG[2]
       #DQDZ = GMLZ.dot(REFG[2])
       DZDX = REFS[15]
       #DZDX2 = REFS[16]
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
                     
       def computeUpdate(coeff, solA, sol2Update, firstStage):
              
              # Compute 1st derivatives
              if firstStage:
                     DqDx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_CFD, DDZM_CFD)
              else:
                     DqDx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_CPU, DDZM_CPU)
              
              # Compute dynamics update
              rhsDyn = computeRHSUpdate_dynamics(solA, DqDx, DqDz)
              
              # Update diffusion coefficients here at the FIRST stage only
              if firstStage:
                     UD = solA[:,0] + init0[:,0]
                     WD = solA[:,1]
                     
                     # Compute DynSGS or Flow Dependent diffusion coefficients
                     QM = bn.nanmax(np.abs(solA), axis=0)
                     filtType = 'maximum'
                     newDiff = rescf.computeResidualViscCoeffs(DIMS, rhsDyn, QM, UD, WD, DX2, DZ2, DXZ, filtType)
                     
                     DCF[0][:,0] = newDiff[0]
                     DCF[1][:,0] = newDiff[1]
                     del(newDiff)
                     firstStage = False
                     
              # Compute 2nd derivatives
              P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz = \
                     tendency.computeFieldDerivatives2(DqDx, DqDz, REFG, DDXM_CFD, DDZM_CFD, DZDX)
                     
              #P2qPx2, P2qPz2, P2qPzx, P2qPxz, PqPx, PqPz = \
              #       tendency.computeFieldDerivativesFlux(DqDx, DqDz, DCF, REFG, DDXM_CFD, DDZM_CFD, DZDX)
              
              # Compute the diffusion update
              rhsDif = computeRHSUpdate_diffusion(solA, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz)
              
              # Dynamics and diffusive update
              rhsUpdate = rhsDyn + rhsDif
              
              # Apply update
              dsol = coeff * DT * rhsUpdate
              solB = sol2Update + dsol
              
              # Enforce essential boundary conditions
              solB[zeroDex[0],0] = np.zeros(len(zeroDex[0]))
              solB[zeroDex[1],1] = np.zeros(len(zeroDex[1]))
              solB[zeroDex[2],2] = np.zeros(len(zeroDex[2]))
              solB[zeroDex[3],3] = np.zeros(len(zeroDex[3]))
              U = solB[:,0] + init0[:,0]
              solB[ebcDex[1],1] = np.array(DZDX[ebcDex[1],0] * U[ebcDex[1]])
              
              #''' TURNED ON IN ORIGINAL RUN
              # Apply Rayleigh layer implicitly
              propagator = np.reciprocal(1.0 + (mu * coeff * DT) * RLM.data)
              solB = propagator.T * solB
              #'''
              
              return solB
       
       def computeRHSUpdate_dynamics(fields, DqDx, DqDz):
              U = fields[:,0] + init0[:,0]
              W = fields[:,1]
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, DqDx, DqDz, REFG, DZDX, RdT_bar, fields, U, W, ebcDex, zeroDex)
                     
              return rhs
       
       def computeRHSUpdate_diffusion(fields, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz):
              
              rhs = tendency.computeDiffusionTendency(PHYS, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                                      REFS, ebcDex, zeroDex, DX2, DZ2, DXZ, DCF, DynSGS)
                     
              #rhs = tendency.computeDiffusiveFluxTendency(PHYS, PqPx, PqPz, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
              #                                        REFS, ebcDex, zeroDex, DX2, DZ2, DXZ, DynSGS)
       
              return rhs
       
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
              sol1 = computeUpdate(0.377268915331368, sol, sol, True)
              # Stage 2
              sol2 = computeUpdate(0.377268915331368, sol1, sol1, False)
              # Stage 3
              sol3 = np.array(0.426988976571684 * sol + 0.5730110234283154 * sol2)
              sol2 = computeUpdate(0.216179247281718, sol2, sol3, False)
              # Stage 4
              sol3 = np.array(0.193245318771018 * sol + 0.199385926238509 * sol1 + 0.607368754990473 * sol2)
              sol2 = computeUpdate(0.229141351401419, sol2, sol3, False)
              # Stage 5
              sol3 = np.array(0.108173740702208 * sol1 + 0.891826259297792 * sol2)
              sol = computeUpdate(0.336458325509300, sol2, sol3, False)
              
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
              
              sol = computeUpdate(c1, sol, sol, True)
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
              #solB, rhsDyn, resDyn = ketcheson93(sol0)
              #solB, rhsDyn, resDyn = RK64_NL(sol0)
              solB = ssprk53_Opt(sol0)
       elif order == 4:
              solB = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketcheson62(sol0)
       
       return solB, DCF