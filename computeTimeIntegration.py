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

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, sol0, INIT, uRamp, zeroDex, extDex, neuDex, botDex, udex, DynSGS, order):
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
       
       def computeRHSUpdate(fields, Dynamics, DynSGS, FlowDiff2):
              if Dynamics:
                     # Scale background wind and shear with ramp up factor
                     U = fields[:,0] + uRamp * INIT[udex]
                     (REFG[4])[:,1] *= uRamp
                     # Compute dynamical tendencies
                     rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DDXM_GML, DDZM_GML, DZDX, RdT_bar, fields, U, neuDex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     rhs[zeroDex[0],0] *= 0.0
                     rhs[zeroDex[1],1] *= 0.0
                     rhs[zeroDex[2],2] *= 0.0
                     rhs[zeroDex[3],3] *= 0.0
              if DynSGS:
                     rhs = tendency.computeDiffusiveFluxTendency(RESCF, DDXM, DDZM, DZDX, fields, neuDex)
                     #rhs = tendency.computeDiffusionTendency(RESCF, DDXM, DDZM, DZDX, fields, udex, wdex, pdex, tdex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     #rhs[extDex[0],0] *= 0.0
                     #rhs[extDex[1],1] *= 0.0
                     #rhs[extDex[2],2] *= 0.0
                     #rhs[extDex[3],3] *= 0.0
                     rhs[zeroDex[0],0] *= 0.0
                     rhs[zeroDex[1],1] *= 0.0
                     rhs[zeroDex[2],2] *= 0.0
                     rhs[zeroDex[3],3] *= 0.0
              if FlowDiff2:
                     rhs = tendency.computeDiffusiveFluxTendency(RESCF, DDXM, DDZM, DZDX, fields, neuDex)
                     #rhs = tendency.computeDiffusionTendency(RESCF, DDXM, DDZM, DZDX, fields)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendency at all boundary DOF
                     rhs[extDex[0],0] *= 0.0
                     rhs[extDex[1],1] *= 0.0
                     rhs[extDex[2],2] *= 0.0
                     rhs[extDex[3],3] *= 0.0
                     rhs[zeroDex[0],0] *= 0.0
                     rhs[zeroDex[1],1] *= 0.0
                     rhs[zeroDex[2],2] *= 0.0
                     rhs[zeroDex[3],3] *= 0.0
                     
              return rhs
       
       def computeUpdate(coeff, sol0, rhs):
              #Apply updates
              dsol = coeff * DT * rhs
              sol1 = sol0 + dsol
              U = sol1[:,0] + uRamp * INIT[udex]
              sol1[botDex,1] = np.array(dHdX * U[botDex])
              # Enforce BC's
              #'''
              topDex = sorted(set.difference(set(zeroDex[1]), set(botDex)))
              sol1[zeroDex[0],0] *= 0.0
              sol1[topDex,1] *= 0.0
              sol1[zeroDex[2],2] *= 0.0
              sol1[zeroDex[3],3] *= 0.0
              #'''
              return sol1
       
       def ssprk22(sol, Dynamics, DynSGS, FlowDiff):
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(1.0, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol = computeUpdate(0.5, sol1, rhs)
              sol = np.array(0.5 * (sol + sol1))
              
              return sol, rhs
       
       def ssprk34(sol, Dynamics, DynSGS, FlowDiff):
              # Stage 1
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(0.5, sol, rhs)
              # Stage 2
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol2 = computeUpdate(0.5, sol1, rhs)
              # Stage 3
              sol = np.array(2.0/3.0 * sol + 1.0 / 3.0 * sol2)
              rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              sol1 = computeUpdate(1.0/6.0, sol, rhs)
              # Stage 4
              rhs = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              sol = computeUpdate(0.5, sol1, rhs)
              
              return sol, rhs
       
       def ketchenson93(sol, Dynamics, DynSGS, FlowDiff):
              for ii in range(7):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs)
                     
                     if ii == 1:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9 (diffusion applied here)
              for ii in range(2):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs)
                     
              return sol, rhs
       
       def ketchenson104(sol, Dynamics, DynSGS, FlowDiff):
              sol1 = np.array(sol)
              for ii in range(1,6):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
                     sol = computeUpdate(c1, sol, rhs)
              
              sol1 = np.array(0.04 * sol1 + 0.36 * sol)
              sol = np.array(15.0 * sol1 - 5.0 * sol)
              
              for ii in range(6,10):
                     rhs = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)                     
                     sol = computeUpdate(c1, sol, rhs)
                     
              sol = np.array(sol1 + 0.6 * sol + 0.1 * DT * rhs)
              
              return sol, rhs
       #'''
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              sol, rhsDyn = ketchenson93(sol0, True, False, False)
       elif order == 4:
              sol, rhsDyn = ketchenson104(sol0, True, False, False)
              
       # Compute adaptive diffusion coefficients
       U = np.abs(uRamp * INIT[udex] + sol[:,0])
       W = np.abs(sol[:,1])
       if DynSGS:
              # DynSGS compute coefficients
              #total_sol = sol + np.reshape(INIT, (len(udex), 4), order='F')
              #total_sol -= bn.nanmean(total_sol, axis=0)
              #QM = bn.nanmax(np.abs(total_sol), axis=0)
              QM = bn.nanmax(np.abs(sol), axis=0)
              # Estimate the residual
              RES = 1.0 / DT * (sol - sol0)
              RES -= rhsDyn
              # Compute diffusion coefficients
              RESCF = dcoeffs.computeResidualViscCoeffs(RES, QM, U, W, DX, DZ)
              sol, rhsDiff = ssprk34(sol, False, True, False)
       else: 
              # Flow weighted diffusion (Guerra, 2016)
              RESCF = dcoeffs.computeFlowVelocityCoeffs(U, W, DX, DZ)
              sol, rhsDiff = ssprk34(sol, False, False, True)
      
       return sol, (rhsDyn + rhsDiff)