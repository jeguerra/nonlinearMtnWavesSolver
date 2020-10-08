#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
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

def computeTimeIntegrationNL2(PHYS, REFS, REFG, DX, DZ, DX2, DZ2, TOPT, \
                              sol0, init0, rhs0, zeroDex, ebcDex, \
                              DynSGS, DCF, thisTime, isFirstStep):
       
       DT = TOPT[0]
       rampTimeBound = TOPT[2]
       order = TOPT[3]
       dHdX = REFS[6]
       RdT_bar = REFS[9][0]
       
       # Adjust for time ramping
       time = thisTime
       uf, duf = rampFactor(time, rampTimeBound)
           
       GML = REFG[0]
       DWDZ = (REFG[2])[:,1]
       DWDX = REFG[6]
       D2QDZ2 = REFG[5]
       D2WDX2 = REFG[7][0]
       D2WDXZ = REFG[7][1]
       DZDX = REFS[15]
       #'''
       DZDX2 = REFS[16]
       DZDXbc = DZDX[ebcDex[1],0]
       DZDX2bc = DZDX2[ebcDex[1],0]
       #scale = np.expand_dims(np.sqrt(1.0 + DZDX2bc), 1)
       #scale = np.reciprocal(np.sqrt(1.0 + DZDX2bc))
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
              
              # Evaluate the dynamics first
              DqDx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_CPU, DDZM_CPU)
              rhsDyn = computeRHSUpdate_dynamics(solA, DqDx, DqDz)
              
              # Compute 2nd derivatives
              #DqDx[:,1] += DWDX
              #DqDz[:,1] += DWDZ
              D2qDx2, D2qDz2, D2qDxz = \
                     tendency.computeFieldDerivatives2(DqDx, DqDz, DDXM_CFD, DDZM_CFD, DZDX)
              #'''
              D2qDx2[:,1] += D2WDX2
              #D2qDz2[:,1] += D2QDZ2[:,1]
              D2qDz2 += D2QDZ2
              D2qDxz[:,1] += D2WDXZ
              #'''
              # Compute the diffusion update
              rhsDif = computeRHSUpdate_diffusion(D2qDx2, D2qDz2, D2qDxz, DCF)
              
              # Apply diffusion
              rhsDyn += rhsDif
              
              # Apply update
              dsol = coeff * DT * rhsDyn
              solB = sol2Update + dsol
              
              return solB
       
       def computeRHSUpdate_dynamics(fields, DqDx, DqDz):
              U = fields[:,0] + init0[:,0]
              W = fields[:,1] + init0[:,1]
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DZDX, RdT_bar, fields, U, W, ebcDex)
              rhsRay = tendency.computeRayleighTendency(REFG, fields)

              # Null Rayleigh layer on W (has GML)
              rhs += rhsRay
              
              # GML layer on all W and LnT tendencies
              rhs[:,1] = GML.dot(rhs[:,1])
              rhs[:,3] = GML.dot(rhs[:,3])
              
              # Fix essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
              
              # Update the boundary constraint
              rhs[ebcDex[1],1] = dHdX * rhs[ebcDex[1],0]
                     
              return rhs
       
       def computeRHSUpdate_diffusion(D2qDx2, D2qDz2, D2qDxz, dcoeff):
              
              # Compute diffusive tendencies
              if DynSGS:
                     rhs = tendency.computeDiffusionTendency(PHYS, dcoeff, D2qDx2, D2qDz2, D2qDxz, DZDX, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, ebcDex)
              else:
                     rhs = tendency.computeDiffusionTendency(PHYS, dcoeff, D2qDx2, D2qDz2, D2qDxz, DZDX, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, ebcDex)

              # GML layer on all diffusion tendencies
              rhs = GML.dot(rhs)
              # Adjust diffusion at terrain boundary
              rhs[ebcDex[1],:] *= 0.0
              
              return rhs
       
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
              
              sol = computeUpdate(c1, sol, sol)
              sol1 = np.array(sol)
                     
              for ii in range(4):
                     sol = computeUpdate(c1, sol, sol)
                     
              # Compute stage 6 with linear combination
              sol1 = np.array(0.6 * sol1 + 0.4 * sol)
              sol = computeUpdate(c2, sol, sol1)
              
              for ii in range(3):
                     sol = computeUpdate(c1, sol, sol)
                     
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
              solf = ketcheson62(sol0)
       elif order == 3:
              solf = ketcheson93(sol0)
       elif order == 4:
              solf = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solf, rhsDyn = ketcheson62(sol0)
       
       time += DT
       
       DqDx, DqDz = tendency.computeFieldDerivatives(solf, DDXM_CPU, DDZM_CPU)
       rhsf = computeRHSUpdate_dynamics(solf, DqDx, DqDz)
       #input('End of time step. ' + str(time))
       
       # Compute residual and normalizations
       u = np.abs(solf[:,0])
       w = np.abs(solf[:,1] + init0[:,1])
       # Compute DynSGS or Flow Dependent diffusion coefficients
       if DynSGS:
              QM = bn.nanmax(np.abs(solf - bn.nanmean(solf)), axis=0)
              # Trapezoidal Rule estimate of residual
              resInv = (1.0 / TOPT[0]) * (solf - sol0) - 0.5 * (rhsf + rhs0)
              resInv[ebcDex[3],:] *= 0.0
              DCF = rescf.computeResidualViscCoeffs(resInv, QM, u, w, DX, DZ, DX**2, DZ**2, REFG[4])
       else:
              DCF = rescf.computeFlowVelocityCoeffs(u, w, DX, DZ)
       
       return solf, rhsf, time, DCF