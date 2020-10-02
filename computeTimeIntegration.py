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

def computeTimeIntegrationNL2(PHYS, REFS, REFG, DX, DZ, DX2, DZ2, TOPT, \
                              sol0, init0, rhs0, dcoeff0, zeroDex, ebcDex, \
                              DynSGS, thisTime, isFirstStep):
       
       DT = TOPT[0]
       rampTimeBound = TOPT[2]
       order = TOPT[3]
       dHdX = REFS[6]
       RdT_bar = REFS[9][0]
       
       # Adjust for time ramping
       time = thisTime
       uf, duf = rampFactor(time, rampTimeBound)
           
       GML = REFG[0]
       DQDZ = REFG[2]
       D2QDZ2 = REFG[5]
       DZDX = REFS[15]
       #'''
       DZDX2 = REFS[16]
       DZDXbc = DZDX[ebcDex[1],0]
       DZDX2bc = DZDX2[ebcDex[1],0]
       #scale = np.sqrt(1.0 + DZDX2bc)
       #scale = np.expand_dims(np.sqrt(1.0 + DZDX2bc), 1)
       scale = np.reciprocal(np.sqrt(1.0 + DZDX2bc))
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
       
       #DDXM_SP = REFS[16]
       #DDZM_SP = REFS[17]
       #isAllCPU = True
       '''
       plt.figure()
       plt.plot(DQDZ[ebcDex[1],0]); plt.plot(DQDZ[ebcDex[1]+1,0])
       plt.xlim(150, 300); plt.legend(('dudz at BC', 'dudz at level 1'))
       plt.show()
       plt.figure()
       plt.plot(DQDZ[ebcDex[1],2]); plt.plot(DQDZ[ebcDex[1]+1,2])
       plt.xlim(150, 300); plt.legend(('dpdz at BC', 'dpdz at level 1'))
       plt.show()
       plt.figure()
       plt.plot(DQDZ[ebcDex[1],3]); plt.plot(DQDZ[ebcDex[1]+1,3])
       plt.xlim(150, 300); plt.legend(('dtdz at BC', 'dtdz at level 1'))
       plt.show()
       '''
       def computeUpdate(coeff, solA, sol2Update, rhsDyn0):
              
              # Evaluate the dynamics first
              DqDx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_CPU, DDZM_CPU)
              
              # Compute 2nd derivatives
              D2qDx2, D2qDz2, D2qDxz = \
                     tendency.computeFieldDerivatives2(DqDx, DqDz, DDXM_CFD, DDZM_CFD, DZDX, D2QDZ2)
                     
              rhsDyn = computeRHSUpdate_dynamics(solA, DqDx, DqDz)
              
              # Apply dynamics update
              dsol = coeff * DT * rhsDyn
              solB = sol2Update + dsol
              
              # Compute the diffusion update
              rhsDif = computeRHSUpdate_diffusion(D2qDx2, D2qDz2, D2qDxz, dcoeff0)
              
              # Apply diffusion update
              dsol = coeff * DT * rhsDif
              solB += dsol
              
              return solB, rhsDyn, rhsDif
       
       def computeRHSUpdate_dynamics(fields, DqDx, DqDz):
              U = fields[:,0] + uf * init0[:,0]
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DZDX, RdT_bar, fields, U, uf, ebcDex)
              rhsRay = tendency.computeRayleighTendency(REFG, fields)
              '''
              plt.figure(figsize=(15.0, 10.0))
              plt.subplot(2,2,1)
              plt.plot(DqDx[ebcDex[1],0]); plt.plot(DqDx[ebcDex[1]+1,0])
              plt.xlim(150, 300); plt.legend(('dqdx at BC', 'dqdx at level 1'))
              plt.subplot(2,2,2)
              plt.plot(DqDz[ebcDex[1],0]); plt.plot(DqDz[ebcDex[1]+1,0])
              plt.xlim(150, 300); plt.legend(('dqdz at BC', 'dqdz at level 1'))
              plt.subplot(2,2,3)
              plt.plot(DqDx[ebcDex[1],3]); plt.plot(DqDx[ebcDex[1]+1,3])
              plt.xlim(150, 300); plt.legend(('dqdx at BC', 'dqdx at level 1'))
              plt.subplot(2,2,4)
              plt.plot(DqDz[ebcDex[1],3]); plt.plot(DqDz[ebcDex[1]+1,3])
              plt.xlim(150, 300); plt.legend(('dqdz at BC', 'dqdz at level 1'))
              plt.show()
              '''
              # Null Rayleigh layer on W (has GML)
              rhs += rhsRay
              
              # GML layer on all diffusion tendencies
              rhs[:,1] = GML.dot(rhs[:,1])
              
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
       
              '''
              plt.figure(figsize=(10.0, 5.0))
              plt.subplot(1,2,1)
              plt.plot(dcoeff[0][ebcDex[1],0]); plt.plot(dcoeff[0][ebcDex[1]+1,0])
              plt.xlim(150, 300); plt.legend(('X coeffs at BC', 'X coeffs at level 1'))
              plt.subplot(1,2,2)
              plt.plot(dcoeff[1][ebcDex[1],0]); plt.plot(dcoeff[1][ebcDex[1]+1,0])
              plt.xlim(150, 300); plt.legend(('Z coeffs at BC', 'Z coeffs at level 1'))
              plt.show()
              '''
              '''
              plt.figure(figsize=(15.0, 10.0))
              plt.subplot(2,3,1)
              plt.plot(D2QDZ2[ebcDex[1],0]); plt.plot(DQDZ[ebcDex[1]+1,0])
              plt.xlim(150, 300); plt.legend(('D2UDZ2 at BC', 'D2UDZ2 at level 1'))
              plt.subplot(2,3,2)
              plt.plot(D2QDZ2[ebcDex[1],2]); plt.plot(D2QDZ2[ebcDex[1]+1,2])
              plt.xlim(150, 300); plt.legend(('D2PDZ2 at BC', 'D2PDZ2 at level 1'))
              plt.subplot(2,3,3)
              plt.plot(D2QDZ2[ebcDex[1],3]); plt.plot(D2QDZ2[ebcDex[1]+1,3])
              plt.xlim(150, 300); plt.legend(('D2ThDZ2 at BC', 'D2ThDZ2 at level 1'))
              plt.subplot(2,3,4)
              plt.plot(D2qDx2[ebcDex[1],3]); plt.plot(D2qDx2[ebcDex[1]+1,3])
              plt.xlim(150, 300); plt.legend(('d2qdx2 at BC', 'd2qdx2 at level 1'))
              plt.subplot(2,3,5)
              plt.plot(D2qDz2[ebcDex[1],3]); plt.plot(D2qDz2[ebcDex[1]+1,3])
              plt.xlim(150, 300); plt.legend(('d2qdz2 at BC', 'd2qdz2 at level 1'))
              plt.subplot(2,3,6)
              plt.plot(D2qDxz[ebcDex[1],3]); plt.plot(D2qDxz[ebcDex[1]+1,3])
              plt.xlim(150, 300); plt.legend(('d2qdxz at BC', 'd2qdxz at level 1'))
              plt.show()
              '''
              '''
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
              '''
              # Adjust diffusion at terrain boundary
              #rhs[ebcDex[1],0] = rhs[ebcDex[1]+1,0]
              #rhs[ebcDex[1],1] = dHdX * rhs[ebcDex[1],0]
              #rhs[ebcDex[1],0:2] *= 0.0
              rhs[ebcDex[1],:] *= 0.0
              
              return rhs
       
       def ketcheson62(sol):
              m = 5
              c1 = 1 / (m-1)
              c2 = 1 / m
              sol1 = np.array(sol)
              rhsDyn = rhs0
              for ii in range(m):
                     if ii == m-1:
                            sol1 = c2 * ((m-1) * sol + sol1)
                            sol, rhsDyn, rhsDif = computeUpdate(c2, sol, sol1, rhsDyn)
                     else:
                            sol, rhsDyn, rhsDif = computeUpdate(c1, sol, sol, rhsDyn)
                      
              return sol, rhsDyn, rhsDif
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 15.0
              
              rhsDyn = rhs0
              sol1, rhsDyn, rhsDif = computeUpdate(c1, sol, sol, rhsDyn)
                     
              for ii in range(4):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol, sol, rhsDyn)
                     
              # Compute stage 6 with linear combination
              sol1 = 0.6 * sol1 + 0.4 * sol
              sol, rhsDyn, rhsDif = computeUpdate(c2, sol, sol1, rhsDyn)
              
              for ii in range(3):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol, sol, rhsDyn)
                     
              return sol, rhsDyn, rhsDif
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
       
              sol2 = np.array(sol)
              for ii in range(5):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol, sol)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol, sol)
                     
              sol2Update = sol2 + 0.6 * sol
              sol, rhsDyn, rhsDif = computeUpdate(0.1, sol, sol2Update)
              
              return sol, rhsDyn, rhsDif
       #'''
#       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solf, rhsDyn, rhsDif = ketcheson62(sol0)
       elif order == 3:
              solf, rhsDyn, rhsDif = ketcheson93(sol0)
       elif order == 4:
              solf, rhsDyn, rhsDif = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solf, rhsDyn, rhsDif = ketcheson62(sol0)
       
       time += DT
       
       DqDx, DqDz = tendency.computeFieldDerivatives(solf, DDXM_CPU, DDZM_CPU)
       rhsf = computeRHSUpdate_dynamics(solf, DqDx, DqDz)
       #input('End of time step.', time)
       
       return solf, rhsf, time, uf