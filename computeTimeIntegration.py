#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import bottleneck as bn
import computeEulerEquationsLogPLogT as tendency
import computeResidualViscCoeffs as dcoeffs

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, sol0, init0, uRamp, zeroDex, extDex, botDex, DynSGS, order):
       dHdX = REFS[6]
       
       # Dereference operators
       RdT_bar = REFS[9]
       DDXM_GML = REFS[10]
       DDZM_GML = REFS[11]
       DDXM = REFS[12]
       DDZM = REFS[13]
       DZDX = REFS[15]
       
       UB = uRamp * init0[:,0]
       #RESCF = rescf0
              
       def computeUpdate(coeff, solA, rhs, Dynamics):
              #Apply updates
              dsol = coeff * DT * rhs
              if Dynamics:
                     solB = solA + dsol
                     # Update boundary
                     U = solB[:,0] + UB
                     solB[botDex,1] = np.array(dHdX * U[botDex])
              else:
                     solB = solA + dsol
              
              return solB
       
       def computeRHSUpdate(fields, Dynamics, DynSGS, FlowDiff2):              
              if Dynamics:
                     # Update advective U
                     U = fields[:,0] + UB
                     # Compute first derivatives
                     DqDx, PqPx, DqDz = tendency.computeFieldDerivatives(fields, DDXM_GML, DDZM_GML, DZDX)
                     # Compute dynamical tendencies
                     rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, PqPx, DqDz, RdT_bar, fields, U)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     if DynSGS:
                            rhsDiff = tendency.computeDiffusiveFluxTendency(RESCF, DqDx, PqPx, DqDz, DDXM, DDZM, DZDX, fields)
                            rhsDiff[extDex,:] *= 0.0
                            rhs += rhsDiff
                     elif FlowDiff2:
                            rhsDiff = tendency.computeDiffusionTendency(RESCF, DqDx, PqPx, DqDz, DDXM, DDZM, DZDX, fields)
                            rhsDiff[extDex,:] *= 0.0
                            rhs += rhsDiff

              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
                     
              return rhs
       
       def ssprk34(sol, Dynamics, DynSGS, FlowDiff):
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.5, sol, rhs, Dynamics)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.5, sol1, rhs, Dynamics)
              # Stage 3
              sol = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(1.0 / 6.0, sol, rhs, Dynamics)
              # Stage 4
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol = computeUpdate(0.5, sol1, rhs, Dynamics)
              
              return sol, rhs
       
       def ssprk53_Opt(sol, Dynamics, DynSGS, FlowDiff):
              # Optimized truncation error to SSP coefficient method from Higueras, 2019
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.377268915331368, sol, rhs, Dynamics)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.377268915331368, sol1, rhs, Dynamics)
              # Stage 3
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.426988976571684 * sol + 0.5730110234283154 * sol2)
              sol2 = computeUpdate(0.216179247281718, sol2, rhs, Dynamics)
              # Stage 4
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.193245318771018 * sol + 0.199385926238509 * sol1 + 0.607368754990473 * sol2)
              sol2 = computeUpdate(0.229141351401419, sol2, rhs, Dynamics)
              # Stage 5
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.108173740702208 * sol1 + 0.891826259297792 * sol2)
              sol = computeUpdate(0.336458325509300, sol2, rhs, Dynamics)
              
              return sol, rhs
       
       def ketcheson93(sol, Dynamics, DynSGS, FlowDiff, getLocalError):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
              if getLocalError:
                     # Set storage for 2nd order solution
                     sol2 = np.array(sol)
                     
              for ii in range(6):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs, Dynamics)
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                            
              if getLocalError:
                     # Compute the 2nd order update if requested
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol2 = 1.0 / 7.0 * sol2 + 6.0 / 7.0 * sol
                     sol2 = computeUpdate(1.0 / 42.0, sol2, rhs, Dynamics)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs, Dynamics)
              
              if getLocalError:
                     # Output the local error estimate if requested
                     err = sol - sol2
                     return sol, rhs, err
              else:
                     return sol, rhs
       
       def ketcheson104(sol, Dynamics, DynSGS, FlowDiff):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              sol1 = np.array(sol)
              sol2 = np.array(sol)
              for ii in range(5):
                     rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
                     sol1 = computeUpdate(c1, sol1, rhs, Dynamics)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol1)
              sol1 = np.array(15.0 * sol2 - 5.0 * sol1)
              
              for ii in range(4):
                     rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)                     
                     sol1 = computeUpdate(c1, sol1, rhs, Dynamics)
                     
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)       
              sol = np.array(sol2 + 0.6 * sol1)
              sol = computeUpdate(0.1, sol, rhs, Dynamics)
              
              return sol, rhs
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       #'''
       # Get advective flow velocity components
       U = np.abs(sol0[:,0] + UB)
       W = np.abs(sol0[:,1])
       QM = bn.nanmax(sol0 + init0, axis=0)
       RES = computeRHSUpdate(sol0, True, False, False)
       # Compute diffusion coefficients
       if DynSGS:
              RESCF = dcoeffs.computeResidualViscCoeffs(RES, QM, U, W, DX, DZ)
       else:
              RESCF = dcoeffs.computeFlowVelocityCoeffs(U, W, DX, DZ)
       #'''
       # Compute dynamics update
       if order == 3:
              solf, rhsDyn = ketcheson93(sol0, True, True, False, False)
              #solf, rhsDyn, errf = ketcheson93(sol0, True, False, False, True)
       elif order == 4:
              solf, rhsDyn = ketcheson104(sol0, True, True, False)
              
       rhsOut = computeRHSUpdate(solf, True, False, False)
              
       return solf, rhsOut