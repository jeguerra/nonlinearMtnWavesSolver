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

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, sol0, INIT, uRamp, zeroDex, extDex, botDex, udex, DynSGS, order):
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
       DZDX = REFS[15]
       
       UB = uRamp * INIT[udex]
              
       def computeUpdate(coeff, solA, rhs):
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              
              # Update boundary
              U = solB[:,0] + UB
              solB[botDex,1] = np.array(dHdX * U[botDex])
              
              return solB
       
       def computeRHSUpdate(fields, Dynamics, DynSGS, FlowDiff2):              
              if Dynamics:
                     # Update advective U
                     U = fields[:,0] + UB
                     # Compute dynamical tendencies
                     rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DDXM_GML, DDZM_GML, DZDX, RdT_bar, fields, U)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     rhs[zeroDex[0],0] *= 0.0
                     rhs[zeroDex[1],1] *= 0.0
                     rhs[zeroDex[2],2] *= 0.0
                     rhs[zeroDex[3],3] *= 0.0
              if DynSGS:
                     rhs = tendency.computeDiffusiveFluxTendency(RESCF, DDXM, DDZM, DZDX, fields, extDex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendency at all boundary DOF
                     rhs[extDex,:] *= 0.0
              if FlowDiff2:
                     rhs = tendency.computeDiffusionTendency(RESCF, DDXM, DDZM, DZDX, fields, extDex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendency at all boundary DOF
                     rhs[extDex,:] *= 0.0
                     
              return rhs
       
       def ssprk34(sol, Dynamics, DynSGS, FlowDiff):
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.5, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.5, sol1, rhs)
              # Stage 3
              sol = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(1.0 / 6.0, sol, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol = computeUpdate(0.5, sol1, rhs)
              
              return sol, rhs
       
       def kinnGray53(sol, Dynamics, DynSGS, FlowDiff):
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.2, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.2, sol, rhs)
              # Stage 3
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(1.0 / 3.0, sol, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(2.0 / 3.0, sol, rhs)
              # Stage 5
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(-0.25 * sol + 1.25 * sol1)
              sol = computeUpdate(0.75, sol2, rhs)
              
              return sol, rhs
              
       
       def ssprk53_1(sol, Dynamics, DynSGS, FlowDiff):
              # Lowest error coefficient method from Higueras, 2019
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.377268915331368, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.377268915331368, sol1, rhs)
              # Stage 3
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.568606169888847 * sol + 0.4313938301111528 * sol2)
              sol2 = computeUpdate(0.162751482366679, sol2, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.088778858640267 * sol + 0.911221141359733 * sol2)
              sol2 = computeUpdate(0.343775411627798, sol2, rhs)
              # Stage 5
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.210416684957724 * sol1 + 0.789583315042277 * sol2)
              sol = computeUpdate(0.297885240829746, sol2, rhs)
              
              return sol, rhs
       
       def ssprk53_2(sol, Dynamics, DynSGS, FlowDiff):
              # Largest stability region method from Higueras, 2019
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.465388589249323, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.465388589249323, sol1, rhs)
              # Stage 3
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol1 = np.array(0.682342861037239 * sol + 0.317657138962761 * sol1)
              sol1 = computeUpdate(0.12474597313998, sol1, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.465388589249323, sol1, rhs)
              # Stage 5
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol1 = np.array(0.045230974482400 * sol + 0.954769025517600 * sol1)
              sol = computeUpdate(0.154263303748666, sol1, rhs)
              
              return sol, rhs
       
       def ssprk53_Opt(sol, Dynamics, DynSGS, FlowDiff):
              # Optimized truncation error to SSP coefficient method from Higueras, 2019
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.377268915331368, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.377268915331368, sol1, rhs)
              # Stage 3
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.426988976571684 * sol + 0.5730110234283154 * sol2)
              sol2 = computeUpdate(0.216179247281718, sol2, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.193245318771018 * sol + 0.199385926238509 * sol1 + 0.607368754990473 * sol2)
              sol2 = computeUpdate(0.229141351401419, sol2, rhs)
              # Stage 5
              rhs = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              sol2 = np.array(0.108173740702208 * sol1 + 0.891826259297792 * sol2)
              sol = computeUpdate(0.336458325509300, sol2, rhs)
              
              return sol, rhs
       
       def ketcheson93(sol, Dynamics, DynSGS, FlowDiff):
              # Ketchenson, 2008 10.1137/07070485X
              for ii in range(7):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs)
                     
                     if ii == 1:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9
              for ii in range(2):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs)
                     
              return sol, rhs
       
       def ketcheson104(sol, Dynamics, DynSGS, FlowDiff):
              # Ketchenson, 2008 10.1137/07070485X
              sol1 = np.array(sol)
              sol2 = np.array(sol)
              for ii in range(5):
                     rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
                     sol1 = computeUpdate(c1, sol1, rhs)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol1)
              sol1 = np.array(15.0 * sol2 - 5.0 * sol1)
              
              for ii in range(4):
                     rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)                     
                     sol1 = computeUpdate(c1, sol1, rhs)
                     
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)       
              sol = np.array(sol2 + 0.6 * sol1)
              sol = computeUpdate(0.1, sol, rhs)
              
              return sol, rhs
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              solf, rhsDyn = ketcheson93(sol0, True, False, False)
       elif order == 4:
              solf, rhsDyn = ketcheson104(sol0, True, False, False)
              
       # Get advective flow velocity components
       U = np.abs(solf[:,0] + UB)
       W = np.abs(solf[:,1])
       if DynSGS:
              # DynSGS compute coefficients
              QM = bn.nanmax(np.abs(solf), axis=0)
              # Estimate the residual
              RES = computeRHSUpdate(solf, True, False, False)
              # Compute diffusion coefficients
              RESCF = dcoeffs.computeResidualViscCoeffs(RES, QM, U, W, DX, DZ)
              
              # Compute a step of diffusion
              solf, rhsDiff = ssprk34(solf, False, True, False)
       else: 
              # Flow weighted diffusion (Guerra, 2016)
              RESCF = dcoeffs.computeFlowVelocityCoeffs(U, W, DX, DZ)
              solf, rhsDiff = ssprk34(solf, False, False, True)
       
       # Compute final RHS and return
       rhsOut = computeRHSUpdate(solf, True, False, False)
       
       return solf, rhsOut