#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import bottleneck as bn
import computeEulerEquationsLogPLogT as tendency
from computeResidualViscCoeffs import computeResidualViscCoeffs

def computeTimeIntegrationLN(PHYS, REFS, REFG, bN, AN, DX, DZ, DT, RHS, SOLT, INIT, sysDex, udex, wdex, pdex, tdex, botDex, topdex, DynSGS): 
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       sol = SOLT[sysDex,0]
       rhs = RHS[sysDex]
       dHdX = REFS[6]
       
       def computeRHSUpdate():
              rhs = bN - AN.dot(sol)
                 
              return rhs
       
       def computeUpdate(coeff, sol, rhs):
              dsol = coeff * DT * rhs
              sol += dsol
              sol[botDex,1] += dHdX * dsol[botDex,0]
              
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       sol = np.array(SOLT[:,0])
       for ii in range(7):
              sol = computeUpdate(c1, sol, RHS)
              RHS = computeRHSUpdate()
              
              if ii == 1:
                     SOLT[:,:,1] = sol
              
       # Compute stage 6 with linear combination
       sol = np.array(c2 * (3.0 * SOLT[:,:,1] + 2.0 * sol))
       
       # Compute stages 7 - 9 (diffusion applied here)
       for ii in range(2):
              sol = computeUpdate(c1, sol, RHS)
              RHS = computeRHSUpdate()
              
       return sol, rhs

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, sol0, INIT, zeroDex, extDex, botDex, udex, wdex, pdex, tdex, DynSGS, order):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       dHdX = REFS[6]
       
       # Dereference operators
       RdT_bar = REFS[9]
       DDXM_GML = REFS[10]
       DDZM_GML = REFS[11]
       DDXM = REFS[12]
       DDZM = REFS[13]
       DZDX = REFS[16]
              
       def computeDynSGSUpdate(fields):
              if DynSGS:
                     rhsSGS = tendency.computeDynSGSTendency(RESCF, DDXM, DDZM, DZDX, fields, udex, wdex, pdex, tdex)
                     # Null tendency at all boundary DOF
                     rhsSGS[extDex[0],0] *= 0.0
                     rhsSGS[extDex[1],1] *= 0.0
                     rhsSGS[extDex[2],2] *= 0.0
                     rhsSGS[extDex[3],3] *= 0.0
              else:
                     rhsSGS = 0.0
                     
              return rhsSGS
       
       def computeRHSUpdate(fields, U):
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DDXM_GML, DDZM_GML, DZDX, RdT_bar, fields, U)
              rhs += tendency.computeRayleighTendency(REFG, fields)
              # Null tendencies at essential boundary DOF
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0

              return rhs
       
       def computeUpdate(coeff, sol, rhs):
              #Apply updates
              dsol = coeff * DT * rhs
              sol += dsol
              sol[botDex,1] += dHdX * dsol[botDex,0]
              
              return sol
       
       def ssprk22(sol, isEuler, isDynSGS):
              if isEuler:
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U)
              if isDynSGS:
                     rhs = computeDynSGSUpdate(sol)
              
              sol1 = computeUpdate(1.0, sol, rhs)
              
              if isEuler:
                     U = tendency.computeWeightFields(REFS, sol1, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol1, U)
              if isDynSGS:
                     rhs = computeDynSGSUpdate(sol1)
              
              sol = computeUpdate(0.5, sol1, rhs)
              
              sol = np.array(0.5 * (sol + sol1))
              
              return sol, rhs
       
       def ssprk34(sol, isEuler, isDynSGS):
              # Stage 1
              if isEuler:
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U)
              if isDynSGS:
                     rhs = computeDynSGSUpdate(sol)
              
              sol1 = computeUpdate(0.5, sol, rhs)
              # Stage 2
              if isEuler:
                     U = tendency.computeWeightFields(REFS, sol1, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol1, U)
              if isDynSGS:
                     rhs = computeDynSGSUpdate(sol1)
              
              sol2 = computeUpdate(0.5, sol1, rhs)
              # Stage 3
              sol = np.array(2.0/3.0 * sol + 1.0 / 3.0 * sol2)
              if isEuler:
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U)
              if isDynSGS:
                     rhs = computeDynSGSUpdate(sol)
              
              sol1 = computeUpdate(1.0/6.0, sol, rhs)
              # Stage 4
              if isEuler:
                     U = tendency.computeWeightFields(REFS, sol1, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol1, U)
              if isDynSGS:
                     rhs = computeDynSGSUpdate(sol1)
              
              sol = computeUpdate(0.5, sol1, rhs)
              
              return sol, rhs
       
       def ketchenson93(sol, isEuler, isDynSGS):
              for ii in range(7):
                     if isEuler:
                            U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                            rhs = computeRHSUpdate(sol, U)
                     if isDynSGS:
                            rhs = computeDynSGSUpdate(sol)
                     
                     sol = computeUpdate(c1, sol, rhs)
                     
                     if ii == 1:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9 (diffusion applied here)
              for ii in range(2):
                     if isEuler:
                            U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                            rhs = computeRHSUpdate(sol, U)
                     if isDynSGS:       
                            rhs = computeDynSGSUpdate(sol)
                     
                     sol = computeUpdate(c1, sol, rhs)
                     
              return sol, rhs
       
       def ketchenson104(sol, isEuler, isDynSGS):
              sol1 = np.array(sol)
              for ii in range(1,6):
                     if isEuler:
                            U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                            rhs = computeRHSUpdate(sol, U)
                     if isDynSGS:
                            rhs = computeDynSGSUpdate(sol)
                     
                     sol = computeUpdate(c1, sol, rhs)
              
              sol1 = np.array(0.04 * sol1 + 0.36 * sol)
              sol = np.array(15.0 * sol1 - 5.0 * sol)
              
              for ii in range(6,10):
                     if isEuler:
                            U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                            rhs = computeRHSUpdate(sol, U)
                     if isDynSGS:
                            rhs = computeDynSGSUpdate(sol)
                     
                     sol = computeUpdate(c1, sol, rhs)
                     
              sol = np.array(sol1 + 0.6 * sol + 0.1 * DT * rhs)
              
              return sol, rhs
       #'''
       #%% THE KETCHENSON SSP(9,3) METHOD
       if order == 3:
              # Compute dynamics update
              sol, rhs = ketchenson93(sol0, True, False)
              # Compute diffusion update
              if DynSGS:
                     QM = bn.nanmax(sol, axis=0)
                     RES = 1.0 / DT * (sol - sol0) + rhs
                     RESCF = computeResidualViscCoeffs(RES, QM, DX, DZ)
                     #sol, rhs = ketchenson93(sol, False, True)
                     sol, rhs = ssprk34(sol, False, True)
       
       #%% THE KETCHENSON SSP(10,4) METHOD
       elif order == 4:
              # Compute dynamics update
              sol, rhs = ketchenson104(sol0, True, False)
              # Compute diffusion update
              if DynSGS:
                     QM = bn.nanmax(sol, axis=0)
                     RES = 1.0 / DT * (sol - sol0) + rhs
                     RESCF = computeResidualViscCoeffs(RES, QM, DX, DZ)
                     sol, rhs = ssprk34(sol, False, True)
              
       return sol, rhs