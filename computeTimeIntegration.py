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

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, sol0, init0, rhs0, res0, dcoeff0, zeroDex, extDex, botDex, DynSGS, order, vsnd0, thisTime, rampTime):
       dHdX = REFS[6]
       
       # Dereference operators
       RdT_bar = REFS[9]
       DDXM_GML = REFS[10]
       DDZM_GML = REFS[11]
       #DDXM = REFS[12]
       #DDZM = REFS[13]
       DZDX = REFS[15]
       
       # Set the time dependent background quantities at the top of the step
       if thisTime == 0.0:
              uRamp = 0.0
       elif thisTime <= rampTime:
              uRamp = 0.5 * (1.0 - mt.cos(mt.pi / rampTime * thisTime))
       else:
              uRamp = 1.0
       
       UZ = uRamp * init0[:,0]
       DUDZ = uRamp * (REFG[4])[:,0]
       time = thisTime
       
       def computeUpdate(coeff, time, solA, rhs, rhsInv0, rhsInv, u0, dudz0):
              U = u0
              DUDZ = dudz0
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              # Update time dependent background wind field
              time += coeff * DT
              if time <= rampTime:
                     uRamp = 0.5 * (1.0 - mt.cos(mt.pi / rampTime * time))
              else:
                     uRamp = 1.0
                     
              U = uRamp * init0[:,0]
              DUDZ = uRamp * (REFG[4])[:,0]
              
              # Update boundary
              solB[botDex,1] = dHdX * (U[botDex] + solB[botDex,0])
              #print(time, uRamp, init0[0,0], U[0], np.max(solB[botDex,1]))
              
              #'''
              # Update diffusion coefficients
              UD = np.abs(solB[:,0] + U)
              WD = np.abs(solB[:,1])
              QM = bn.nanmax(solB, axis=0)
              resInv = 1.0 / DT * (solB - solA) - 0.5 * (rhsInv0 + rhsInv)
              
              #'''
              if DynSGS:
                     dcoeff = dcoeffs.computeResidualViscCoeffs(resInv, QM, UD, WD, DX, DZ, vsnd0)
              else:
                     dcoeff = dcoeffs.computeFlowVelocityCoeffs(UD, WD, DX, DZ, vsnd0)
              #'''
              return time, solB, dcoeff, U, DUDZ
       
       def computeRHSUpdate_inviscid(fields, U):
              (REFG[4])[:,0] = DUDZ
              # Compute first derivatives
              DqDx, DqDz = tendency.computeFieldDerivatives(fields, DDXM_GML, DDZM_GML)
              # Compute dynamical tendencies
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DZDX, RdT_bar, fields, U, botDex)
              rhs += tendency.computeRayleighTendency(REFG, fields)

              # Fix Essential boundary conditions
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
                     
              return rhs
       
       def computeRHSUpdate(fields, dcoeff, U, DUDZ):
              (REFG[4])[:,0] = DUDZ
              # Compute first derivatives
              DqDx, DqDz = tendency.computeFieldDerivatives(fields, DDXM_GML, DDZM_GML)
              # Compute dynamical tendencies
              rhsInv = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DqDx, DqDz, DZDX, RdT_bar, fields, U, botDex)
              rhsInv += tendency.computeRayleighTendency(REFG, fields)
              
              # Fix Essential boundary conditions
              rhsInv[zeroDex[0],0] *= 0.0
              rhsInv[zeroDex[1],1] *= 0.0
              rhsInv[zeroDex[2],2] *= 0.0
              rhsInv[zeroDex[3],3] *= 0.0
              
              # Compute diffusive tendencies
              if DynSGS:
                     #rhsDiff = tendency.computeDiffusionTendency(dcoeff, DqDx, DqDz, DDXM_GML, DDZM_GML, DZDX, fields)
                     rhsDiff = tendency.computeDiffusiveFluxTendency(dcoeff, DqDx, DqDz, DDXM_GML, DDZM_GML, DZDX, fields)
              else:
                     rhsDiff = tendency.computeDiffusionTendency(dcoeff, DqDx, DqDz, DDXM_GML, DDZM_GML, DZDX, fields)
              
              # Fix Essential boundary conditions
              #rhsDiff[zeroDex[0],0] *= 0.0
              #rhsDiff[zeroDex[1],1] *= 0.0
              #rhsDiff[zeroDex[2],2] *= 0.0
              #rhsDiff[zeroDex[3],3] *= 0.0
              # Fix on all boundary edges
              rhsDiff[extDex,0] *= 0.0
              rhsDiff[extDex,1] *= 0.0
              rhsDiff[extDex,2] *= 0.0
              rhsDiff[extDex,3] *= 0.0
              rhs = rhsInv + rhsDiff

              return rhs, rhsInv
       
       def ketcheson93(time0, sol, rhsInv0, dcoeff0, u0, dudz0, getLocalError):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 5.0
              U = u0
              DUDZ = dudz0
              dc = dcoeff0
              rhs0 = rhsInv0
              time = time0
              if getLocalError:
                     # Set storage for 2nd order solution
                     sol2 = np.array(sol)
                     
              for ii in range(6):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     time, sol, dc, U, DUDZ = computeUpdate(c1, time, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
                     
                     if ii == 0:
                            sol1 = np.array(sol)
                            
              if getLocalError:
                     # Compute the 2nd order update if requested
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     sol2 = 1.0 / 7.0 * sol2 + 6.0 / 7.0 * sol
                     time, sol2, dc, U, DUDZ = computeUpdate(1.0 / 42.0, time, sol2, rhs, rhs0, rhsInv, U, DUDZ)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(3):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     time, sol, dc, U, DUDZ = computeUpdate(c1, time, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
              
              if getLocalError:
                     # Output the local error estimate if requested
                     err = sol - sol2
                     return sol, time, U, err
              else:
                     return sol, time, U
       
       def ketcheson104(time0, sol, rhsInv0, dcoeff0, u0, dudz0):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              U = u0
              DUDZ = dudz0
              dc = dcoeff0
              rhs0 = rhsInv0
              time = time0
              sol1 = np.array(sol)
              sol2 = np.array(sol)
              for ii in range(5):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     time, sol, dc, U, DUDZ = computeUpdate(c1, time, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol1)
              sol1 = np.array(15.0 * sol2 - 5.0 * sol1)
              
              for ii in range(4):
                     rhs, rhsInv = computeRHSUpdate(sol, dc, U, DUDZ)
                     time, sol, dc, U, DUDZ = computeUpdate(c1, time, sol, rhs, rhs0, rhsInv, U, DUDZ)
                     rhs0 = rhsInv
                     
              rhs, rhsInv = computeRHSUpdate(sol1, dc, U, DUDZ)       
              sol = np.array(sol2 + 0.6 * sol1)
              time, sol, dc, U, DUDZ = computeUpdate(0.1, time, sol, rhs, rhs0, rhsInv, U, DUDZ)
              rhs0 = rhsInv
              
              return sol, time, U
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              solf, time, UB = ketcheson93(time, sol0, rhs0, dcoeff0, UZ, DUDZ, False)
              #solf, rhsDyn, errf = ketcheson93(sol0, True, True)
       elif order == 4:
              solf, time, UB = ketcheson104(time, sol0, rhs0, UZ, DUDZ, dcoeff0)
              
       #'''       
       # Get advective flow velocity components
       U = np.abs(solf[:,0] + UB)
       W = np.abs(solf[:,1])
       rhsOut = computeRHSUpdate_inviscid(solf, U)
       # Estimate residual
       resOut = 1.0 / DT * (solf - sol0) - 0.5 * (rhs0 + rhsOut)
       # Compute diffusion coefficients
       QM = bn.nanmax(solf, axis=0)
       dcoeff_fld = dcoeffs.computeFlowVelocityCoeffs(U, W, DX, DZ, vsnd0)
       dcoeff_sgs = dcoeffs.computeResidualViscCoeffs(resOut, QM, U, W, DX, DZ, vsnd0)
       
       return solf, rhsOut, resOut, dcoeff_fld, dcoeff_sgs, time