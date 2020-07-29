#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import computeEulerEquationsLogPLogT as tendency

def computeTimeIntegrationVISC(PHYS, REFS, REFG, TOPT, sol0, init0, dcoeff0, zeroDex, botDex, ebcDex, isFirstStep):
       
       DT = TOPT[0]
       order = TOPT[3]
       dHdX = REFS[6]
       
       GMLX = REFG[0]
       GMLZ = REFG[1]
       
       if isFirstStep:
              DDXM = REFS[10]
              DDZM = REFS[11]
       else:
              DDXM = REFS[12]
              DDZM = REFS[13]
              
       DZDX = REFS[15]
       
       #DDXM_SP = REFS[16]
       #DDZM_SP = REFS[17]
       
       UZ = init0[:,0]
       U = sol0[:,0] + UZ
       DCF = dcoeff0
       
       def computeUpdate(coeff, solA, rhs):
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              # Update boundary
              U = solB[:,0] + UZ
              solB[botDex,1] = dHdX * U[botDex]
              
              return solB
       
       def computeRHSUpdate(fields):
              # Compute first derivatives
              DqDx, DqDz, DqDx_GML, DqDz_GML = \
                     tendency.computeFieldDerivatives(fields, DDXM, DDZM, GMLX, GMLZ)
              
              rhs = tendency.computeDiffusionTendency(DCF, DqDx_GML, DqDz_GML, DDXM, DDZM, DZDX, fields, ebcDex)
              #rhs = tendency.computeDiffusiveFluxTendency(DCF, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
       
              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
              
              return rhs
       
       def ssprk34(sol):
              # Stage 1
              rhs = computeRHSUpdate(sol)
              sol1 = computeUpdate(0.5, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1)
              sol2 = computeUpdate(0.5, sol1, rhs)
              # Stage 3
              sol = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              rhs = computeRHSUpdate(sol2)
              sol1 = computeUpdate(1.0 / 6.0, sol, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol1)
              sol = computeUpdate(0.5, sol1, rhs)
              
              return sol
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
                     
              for ii in range(6):
                     rhs = computeRHSUpdate(sol)
                     sol = computeUpdate(c1, sol, rhs)
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs = computeRHSUpdate(sol)
                     sol = computeUpdate(c1, sol, rhs)
              
              return sol
       #'''
       
       #%% THE MAIN TIME INTEGRATION STAGES       
       # Compute dynamics update
       solf = ketcheson93(sol0)
       
       return solf