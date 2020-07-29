#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
import bottleneck as bn
import computeEulerEquationsLogPLogT as tendency
import computeResidualViscCoeffs as dcoeffs

def rampFactor(time, timeBound):
       if time == 0.0:
              uRamp = 0.0
       elif time <= timeBound:
              uRamp = 0.5 * (1.0 - mt.cos(mt.pi / timeBound * time))
              #uRamp = mt.sin(0.5 * mt.pi / timeBound * time)
              #uRamp = uRamp**4
       else:
              uRamp = 1.0
              
       return uRamp

def computeTimeIntegrationNL2(PHYS, REFS, REFG, DX, DZ, DX2, DZ2, TOPT, sol0, init0, dcoeff0, zeroDex, ebcDex, botDex, DynSGS, thisTime, isFirstStep):
       
       DT = TOPT[0]
       rampTime = TOPT[2]
       order = TOPT[3]
       dHdX = REFS[6]
       
       # Dereference operators
       GMLX = REFG[0]
       GMLZ = REFG[1]
       RLM = REFG[5]
       
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
       
       dcoeff = dcoeff0
       time = thisTime
       
       def computeUpdate(coeff, solA, rhs, rhsInv):
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              # Update boundary
              U = solB[:,0] + init0[:,0]
              solB[botDex,1] = dHdX * U[botDex]
              
              #'''
              # Update diffusion coefficients
              UD = np.abs(solB[:,0] + U)
              WD = np.abs(solB[:,1])
              if DynSGS:
                     QM = bn.nanmax(solB, axis=0)
                     # Make a 1st order estimate of the residual
                     resInv = 1.0 / (coeff * DT) * (solB - solA) - rhsInv
                     dcoeff = dcoeffs.computeResidualViscCoeffs(resInv, QM, UD, WD, DX, DZ, DX2, DZ2, RLM)
              else:
                     dcoeff = dcoeffs.computeFlowVelocityCoeffs(UD, WD, DX, DZ)
              
              return solB, dcoeff
       
       def computeRHSUpdate_inviscid(fields):
              U = fields[:,0] + init0[:,0]
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
       
       def computeRHSUpdate(fields):
              U = fields[:,0] + init0[:,0]
              # Compute first derivatives
              DqDx, DqDz, DqDx_GML, DqDz_GML = \
                     tendency.computeFieldDerivatives(fields, DDXM, DDZM, GMLX, GMLZ)
              # Compute dynamical tendencies
              rhsInv = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, DZDX, RdT_bar, fields, U)
              rhsInv += tendency.computeRayleighTendency(REFG, fields)
              
              # Compute diffusive tendencies
              if DynSGS:
                     rhsDiff = tendency.computeDiffusionTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
              else:
                     rhsDiff = tendency.computeDiffusionTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
                            
              # Add inviscid and diffusion tendencies
              rhs = rhsInv + rhsDiff
              
              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0

              return rhs, rhsInv
       
       def ketcheson93(sol, getLocalError):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0

              if getLocalError:
                     # Set storage for 2nd order solution
                     sol2 = np.array(sol)
                     
              for ii in range(6):
                     rhs, rhsInv = computeRHSUpdate(sol)
                     sol, dcoeff = computeUpdate(c1, sol, rhs, rhsInv)
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                            
              if getLocalError:
                     # Compute the 2nd order update if requested
                     rhs, rhsInv = computeRHSUpdate(sol)
                     sol2 = 1.0 / 7.0 * sol2 + 6.0 / 7.0 * sol
                     sol2, dcoeff = computeUpdate(1.0 / 42.0, sol2, rhs, rhsInv)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs, rhsInv = computeRHSUpdate(sol)
                     sol, dcoeff = computeUpdate(c1, sol, rhs, rhsInv)
                     
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
                     rhs, rhsInv = computeRHSUpdate(sol)
                     sol , dcoeff= computeUpdate(c1, sol, rhs, rhsInv)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     rhs, rhsInv = computeRHSUpdate(sol)
                     sol, dcoeff = computeUpdate(c1, sol, rhs, rhsInv)
                     
              rhs, rhsInv = computeRHSUpdate(sol)       
              sol = np.array(sol2 + 0.6 * sol)
              sol, dcoeff = computeUpdate(0.1, sol, rhs, rhsInv)
              
              return sol
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              solf = ketcheson93(sol0, False)
       elif order == 4:
              solf = ketcheson104(sol0)
       
       rhsf = computeRHSUpdate_inviscid(solf)
       time += DT
       
       return solf, rhsf, time

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DX2, DZ2, TOPT, sol0, init0, dcoeff0, zeroDex, ebcDex, botDex, DynSGS, thisTime, isFirstStep):
       
       DT = TOPT[0]
       rampTime = TOPT[2]
       order = TOPT[3]
       dHdX = REFS[6]
       
       # Dereference operators
       GMLX = REFG[0]
       GMLZ = REFG[1]
       RLM = REFG[5]
       
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
       
       # Set the time dependent background quantities at the top of the step
       uRamp = rampFactor(thisTime, rampTime)
       
       UZ = uRamp * init0[:,0]
       DUDZ = uRamp * (REFG[3])[:,0]
       time = thisTime
       
       def computeUpdate(coeff, solA, rhs, rhsInv0, rhsInv, u0, dudz0):
              U = u0
              DUDZ = dudz0
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              # Update boundary
              solB[botDex,1] = dHdX * (U[botDex] + solB[botDex,0])
              
              #'''
              # Update diffusion coefficients
              UD = np.abs(solB[:,0] + U)
              WD = np.abs(solB[:,1])
              QM = bn.nanmax(solB, axis=0)
              resInv = (1.0 / (coeff * DT)) * (solB - solA) - 0.5 * (rhsInv0 + rhsInv)
              
              #'''
              if DynSGS:
                     dcoeff = dcoeffs.computeResidualViscCoeffs(resInv, QM, UD, WD, DX, DZ, DX2, DZ2, RLM)
              else:
                     dcoeff = dcoeffs.computeFlowVelocityCoeffs(UD, WD, DX, DZ)
              #'''
              return solB, resInv, dcoeff, U, DUDZ
       
       def computeRHSUpdate_inviscid(fields, U):
              (REFG[3])[:,0] = DUDZ
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
       
       def computeRHSUpdate(fields, dcoeff, U, DUDZ):
              (REFG[3])[:,0] = DUDZ
              # Compute first derivatives
              DqDx, DqDz, DqDx_GML, DqDz_GML = \
                     tendency.computeFieldDerivatives(fields, DDXM, DDZM, GMLX, GMLZ)
              # Compute dynamical tendencies
              rhsInv = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DqDx_GML, DqDz_GML, DZDX, RdT_bar, fields, U)
              rhsInv += tendency.computeRayleighTendency(REFG, fields)
              
              # Compute diffusive tendencies
              if DynSGS:
                     rhsDiff = tendency.computeDiffusionTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
                     #rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
              else:
                     rhsDiff = tendency.computeDiffusionTendency(dcoeff, DqDx, DqDz, DDXM, DDZM, DZDX, fields, ebcDex)
                            
              # Add inviscid and diffusion tendencies
              rhs = rhsInv + rhsDiff
              
              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0

              return rhs, rhsInv
       
       def ketcheson93(sol, rhsInv0, dcoeff0, u0, dudz0, getLocalError):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
              U = u0
              DUDZ = dudz0
              dc = dcoeff0
              rhs0 = rhsInv0
              if getLocalError:
                     # Set storage for 2nd order solution
                     sol2 = np.array(sol)
                     
              for ii in range(6):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     sol, res, dc, U, DUDZ = computeUpdate(c1, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                            
              if getLocalError:
                     # Compute the 2nd order update if requested
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     sol2 = 1.0 / 7.0 * sol2 + 6.0 / 7.0 * sol
                     sol2, res, dc, U, DUDZ = computeUpdate(1.0 / 42.0, sol2, rhs, rhs0, rhsInv, U, DUDZ)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     sol, res, dc, U, DUDZ = computeUpdate(c1, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
              
              if getLocalError:
                     # Output the local error estimate if requested
                     err = sol - sol2
                     return sol, U, dc, err
              else:
                     return sol, U, res, dc
       
       def ketcheson104(sol, rhsInv0, dcoeff0, u0, dudz0):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              U = u0
              DUDZ = dudz0
              dc = dcoeff0
              rhs0 = rhsInv0
              sol2 = np.array(sol)
              for ii in range(5):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     sol, res, dc, U, DUDZ = computeUpdate(c1, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     sol, res, dc, U, DUDZ = computeUpdate(c1, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
                     
              rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)       
              sol = np.array(sol2 + 0.6 * sol)
              sol, res, dc, U, DUDZ = computeUpdate(0.1, sol, rhs, rhs0, rhsInv, U, DUDZ)
              rhs0 = rhsInv
              
              return sol, time, U, res, dc
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       U0 = np.abs(sol0[:,0] + UZ)
       rhs0 = computeRHSUpdate_inviscid(sol0, U0)
       
       # Compute dynamics update
       if order == 3:
              solf, UB, resOut, dcoeff1 = ketcheson93(sol0, rhs0, dcoeff0, UZ, DUDZ, False)
              #solf, rhsDyn, errf = ketcheson93(sol0, True, True)
       elif order == 4:
              solf, UB, resOut, dcoeff1 = ketcheson104(sol0, rhs0, dcoeff0, UZ, DUDZ)
       
       Uf = np.abs(solf[:,0] + UB)
       rhsf = computeRHSUpdate_inviscid(solf, Uf)
       time += DT
       
       return solf, rhsf, dcoeff1, time