#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
import time as timing
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
                              sol0, init0, dcoeff0, zeroDex, ebcDex, \
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
       DZDX = REFS[15]
       #'''
       DZDX2 = REFS[16]
       #D2ZDX2 = REFS[17]
       DZDX2bc = DZDX2[ebcDex[1],0]
       scale = np.reciprocal(np.sqrt(1.0 + DZDX2bc))
       #'''
       if isFirstStep:
              # Use SciPY sparse for dynamics
              DDXM_CPU = REFS[10]
              DDZM_CPU = REFS[11]
              DDXM_GPU = REFS[13][0]
              DDZM_GPU = REFS[13][1]
       else:
              # Use multithreading on CPU and GPU
              DDXM_CPU = REFS[12][0]
              DDZM_CPU = REFS[12][1]
              DDXM_GPU = REFS[13][0]
              DDZM_GPU = REFS[13][1]
       
       #DDXM_SP = REFS[16]
       #DDZM_SP = REFS[17]
       
       isAllCPU = True
       def computeUpdate(coeff, solA):
              
              if isAllCPU:
                     # THIS IS FASTEST WITH CURRENT HARDWARE...
                     DqDx, DqDz, D2qDx2, D2qDz2, D2qDxz = \
                     tendency.computeAllFieldDerivatives_CPU(solA, DDXM_CPU, DDZM_CPU, DZDX, DQDZ)
              else:
                     DqDx, DqDz, D2qDx2, D2qDz2, D2qDxz = \
                            tendency.computeAllFieldDerivatives_CPU2GPU(solA, DDXM_CPU, DDZM_CPU, DDXM_GPU, DDZM_GPU, DZDX, DQDZ)
                     #DqDx, DqDz, D2qDx2, D2qDz2, D2qDxz = \
                     #       tendency.computeAllFieldDerivatives_GPU2CPU(solA, DDXM_CPU, DDZM_CPU, DDXM_GPU, DDZM_GPU, DZDX, DQDZ)
                     #DqDx, DqDz, D2qDx2, D2qDz2, D2qDxz = \
                     #       tendency.computeAllFieldDerivatives_GPU(solA, DDXM_GPU, DDZM_GPU, DZDX, DQDZ)
              
              rhsDyn = computeRHSUpdate_dynamics(solA, DqDx, DqDz)
              rhsDif = computeRHSUpdate_diffusion(D2qDx2, D2qDz2, D2qDxz)
              
              # GML layer only on W
              rhsDyn[:,1] = GML.dot(rhsDyn[:,1])
              
              # GML layer on all diffusion tendencies
              rhsDif = GML.dot(rhsDif)
              # No diffusion at terrain boundary
              rhsDif[ebcDex[1],:] *= 0.0
              # No diffusion at lateral and top
              #rhsDif[ebcDex[0],:] *= 0.0
              #rhsDif[ebcDex[2],:] *= 0.0
              # Scale X component of vector diffusion to local tangent
              #rhsDif[ebcDex[1],0] *= scale
              #rhsDif[ebcDex[1],1] *= scale * DZDXbc
              
              rhs = rhsDyn + rhsDif
              
              # Fix Essential boundary conditions and impose constraint
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0 #dHdX * rhs[ebcDex[1],0]
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
              
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              solB[ebcDex[1],1] += dHdX * dsol[ebcDex[1],0]
              
              return solB, rhsDyn, rhsDif
       
       def computeRHSUpdate_dynamics(fields, DqDx, DqDz):
              U = fields[:,0] + uf * init0[:,0]
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DZDX, RdT_bar, fields, U, uf, ebcDex)
              rhsRay = tendency.computeRayleighTendency(REFG, fields)
              
              # Null Rayleigh layer on W (has GML)
              rhsRay[:,1] *= 0.0
              rhs += rhsRay
                     
              return rhs
       
       def computeRHSUpdate_diffusion(D2qDx2, D2qDz2, D2qDxz):
              # Compute diffusive tendencies
              if DynSGS:
                     rhs = tendency.computeDiffusionTendency(PHYS, dcoeff0, D2qDx2, D2qDz2, D2qDxz, DZDX, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, ebcDex)
              else:
                     rhs = tendency.computeDiffusionTendency(PHYS, dcoeff0, D2qDx2, D2qDz2, D2qDxz, DZDX, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, ebcDex)
       
              return rhs
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
                     
              for ii in range(6):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol)
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              sol[ebcDex[1],1] = dHdX * (uf * init0[ebcDex[1],0] + sol[ebcDex[1],0])
              
              # Compute stages 7 - 9
              for ii in range(3):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol)
                     
              return sol, rhsDyn, rhsDif
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
       
              sol2 = np.array(sol)
              for ii in range(5):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     sol, rhsDyn, rhsDif = computeUpdate(c1, sol)
                     
              sol = np.array(sol2 + 0.6 * sol)
              sol = computeUpdate(0.1, sol)
              
              return sol, rhsDyn, rhsDif
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              solf, rhsDyn, rhsDif = ketcheson93(sol0)
       elif order == 4:
              solf, rhsDyn, rhsDif = ketcheson104(sol0)
       
       time += DT
       '''
       uf, duf = rampFactor(time, rampTimeBound)
       DqDx, DqDz = tendency.computeFieldDerivatives_GPU(solf, DDXM_GPU, DDZM_GPU)
       rhsf = computeRHSUpdate_dynamics(solf, DqDx, DqDz)
       '''
       return solf, rhsDyn, time, uf