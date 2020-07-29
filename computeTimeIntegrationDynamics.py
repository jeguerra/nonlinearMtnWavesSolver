#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import computeEulerEquationsLogPLogT as tendency

def computeTimeIntegrationDYNCS(PHYS, REFS, REFG, TOPT, sol0, init0, zeroDex, botDex, isFirstStep):
       
       DT = TOPT[0]
       order = TOPT[3]
       dHdX = REFS[6]
       
       GMLX = REFG[0]
       GMLZ = REFG[1]
       
       RdT_bar = REFS[9]
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
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, DZDX, RdT_bar, fields, U)
              rhs += tendency.computeRayleighTendency(REFG, fields)
              
              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0

              return rhs
       
       def ketcheson93(sol, getLocalError):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
       
              if getLocalError:
                     # Set storage for 2nd order solution
                     sol2 = np.array(sol)
                     
              for ii in range(6):
                     rhs = computeRHSUpdate(sol)
                     sol = computeUpdate(c1, sol, rhs)
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                            
              if getLocalError:
                     # Compute the 2nd order update if requested
                     rhs = computeRHSUpdate(sol)
                     sol2 = 1.0 / 7.0 * sol2 + 6.0 / 7.0 * sol
                     sol2 = computeUpdate(1.0 / 42.0, sol2, rhs)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs = computeRHSUpdate(sol)
                     sol = computeUpdate(c1, sol, rhs)
              
              if getLocalError:
                     # Output the local error estimate if requested
                     err = sol - sol2
                     return sol, err
              else:
                     return sol
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              sol2 = np.array(sol)
              for ii in range(5):
                     rhs = computeRHSUpdate(sol)
                     sol = computeUpdate(c1, sol, rhs)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     rhs = computeRHSUpdate(sol)
                     sol = computeUpdate(c1, sol, rhs)
                     
              rhs = computeRHSUpdate(sol)       
              sol = np.array(sol2 + 0.6 * sol)
              sol = computeUpdate(0.1, sol, rhs)
              
              return sol
       #'''
       
       #%% THE MAIN TIME INTEGRATION STAGES       
       # Compute dynamics update
       if order == 3:
              solf = ketcheson93(sol0, False)
              #solf, rhsDyn, errf = ketcheson93(sol0, True, True)
       elif order == 4:
              solf = ketcheson104(sol0)
       
       rhsf = computeRHSUpdate(solf)
       
       return solf, rhsf