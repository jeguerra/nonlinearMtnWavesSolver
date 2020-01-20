#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
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

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, RHS, SGS, SOLT, INIT, zeroDex, extDex, botDex, udex, wdex, pdex, tdex, DynSGS, order):
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
                     RESCF = computeResidualViscCoeffs(fields, RHS, DX, DZ, udex, wdex, pdex, tdex)
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
              dsol = coeff * DT * rhs
              sol += dsol
              sol[botDex,1] += dHdX * dsol[botDex,0]
              
              return sol
       #'''
       #%% THE KETCHENSON SSP(9,3) METHOD
       if order == 3:
              # Compute stages 1 - 5
              sol = np.array(SOLT[:,:,0])
              for ii in range(7):
                     sol = computeUpdate(c1, sol, (RHS + SGS))
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     RHS = computeRHSUpdate(sol, U)
                     SGS = computeDynSGSUpdate(sol)
                     
                     if ii == 1:
                            SOLT[:,:,1] = sol
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * SOLT[:,:,1] + 2.0 * sol))
              
              # Compute stages 7 - 9 (diffusion applied here)
              for ii in range(2):
                     sol = computeUpdate(c1, sol, (RHS + SGS))
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     RHS = computeRHSUpdate(sol, U)
                     SGS = computeDynSGSUpdate(sol)
       
       #%% THE KETCHENSON SSP(10,4) METHOD
       elif order == 4:
              SOLT[:,:,1] = SOLT[:,:,0]
              sol = np.array(SOLT[:,:,0])
              for ii in range(1,6):
                     sol = computeUpdate(c1, sol, (RHS + SGS))
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     RHS = computeRHSUpdate(sol, U)
                     SGS = computeDynSGSUpdate(sol)
              
              SOLT[:,:,1] = 0.04 * SOLT[:,:,1] + 0.36 * sol
              sol = 15.0 * SOLT[:,:,1] - 5.0 * sol
              
              for ii in range(6,10):
                     sol = computeUpdate(c1, sol, (RHS + SGS))
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     RHS = computeRHSUpdate(sol, U)
                     SGS = computeDynSGSUpdate(sol)
                     
              sol = SOLT[:,:,1] + 0.6 * sol + 0.1 * DT * RHS
              
       return sol, RHS, SGS