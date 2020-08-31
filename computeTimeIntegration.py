#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
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
       rampTime = TOPT[2]
       order = TOPT[3]
       dHdX = REFS[6]
       
       # Dereference operators
       GMLX = REFG[0]
       GMLZ = REFG[1]
       RLM = REFG[5]
       
       RdT_bar = REFS[9][0]
       SVOL_bar = REFS[9][1]
       if isFirstStep:
              DDXM = REFS[10]
              DDZM = REFS[11]
       else:
              DDXM = REFS[12]
              DDZM = REFS[13]
              
       DZDX = REFS[15]
       DZDX2 = REFS[16]
       D2ZDX2 = REFS[17]
       
       #DDXM_SP = REFS[16]
       #DDZM_SP = REFS[17]
       
       time = thisTime
       timeBound = 300.0
       uf, duf = rampFactor(time, timeBound)
       
       def computeUpdate_dynamics(coeff, solA, rhs):
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              # Update boundary
              solB[ebcDex[1],1] += dHdX * dsol[ebcDex[1],0]
              
              return solB
       
       def computeUpdate_diffusion(coeff, solB, DqDx, DqDz):
              
              rhs = computeRHSUpdate_diffusion(solB, DqDx, DqDz)
              
              #Apply updates
              dsol = coeff * DT * rhs
              solC = solB + dsol
              
              # Update boundary
              solC[ebcDex[1],1] += dHdX * dsol[ebcDex[1],0]
              
              return solC
       
       def computeRHSUpdate_dynamics(fields):
              U = fields[:,0] + init0[:,0]
              # Compute first derivatives of the state
              DqDx, DqDz, DqDx_GML, DqDz_GML = \
                     tendency.computeFieldDerivatives(fields, DDXM, DDZM, GMLX, GMLZ)
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, DZDX, RdT_bar, fields, U, ebcDex[1])
              rhs += tendency.computeRayleighTendency(REFG, fields)

              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
                     
              return rhs, DqDx, DqDz
       
       def computeRHSUpdate_diffusion(fields, DqDx, DqDz):
              # Compute diffusive tendencies
              if DynSGS:
                     rhs = tendency.computeDiffusionTendency(PHYS, dcoeff0, DqDx, DqDz, DDXM, DDZM, DZDX, DZDX2, D2ZDX2, SVOL_bar, fields, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, ebcDex)
              else:
                     rhs = tendency.computeDiffusionTendency(PHYS, dcoeff0, DqDx, DqDz, DDXM, DDZM, DZDX, DZDX2, D2ZDX2, SVOL_bar, fields, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, ebcDex)
              '''
              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
              '''
              return rhs
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
                     
              for ii in range(6):
                     rhs, DqDx, DqDz = computeRHSUpdate_dynamics(sol)
                     sol = computeUpdate_dynamics(c1, sol, rhs)
                     sol = computeUpdate_diffusion(c1, sol, DqDx, DqDz)
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs, DqDx, DqDz = computeRHSUpdate_dynamics(sol)
                     sol = computeUpdate_dynamics(c1, sol, rhs)
                     sol = computeUpdate_diffusion(c1, sol, DqDx, DqDz)
                     
              return sol
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
       
              sol2 = np.array(sol)
              for ii in range(5):
                     rhs, DqDx, DqDz = computeRHSUpdate_dynamics(sol)
                     sol, dcoeff = computeUpdate_dynamics(c1, sol, rhs)
                     sol = computeUpdate_diffusion(c1, sol, DqDx, DqDz)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     rhs, DqDx, DqDz = computeRHSUpdate_dynamics(sol)
                     sol = computeUpdate_dynamics(c1, sol, rhs)
                     sol = computeUpdate_diffusion(c1, sol, DqDx, DqDz)
                     
              rhs = computeRHSUpdate_dynamics(sol)       
              sol = np.array(sol2 + 0.6 * sol)
              sol = computeUpdate_dynamics(0.1, sol, rhs)
              sol = computeUpdate_diffusion(0.1, sol)
              
              return sol
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              solf = ketcheson93(sol0)
       elif order == 4:
              solf = ketcheson104(sol0)
       
       rhsf, DqDx, DqDz = computeRHSUpdate_dynamics(solf)
       time += DT
       #input('STOP: Checking diffusion...')
       
       return solf, rhsf, time