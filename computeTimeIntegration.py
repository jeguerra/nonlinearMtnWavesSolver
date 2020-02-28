#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import bottleneck as bn
#from scipy.integrate import solve_ivp
import computeEulerEquationsLogPLogT as tendency
from computeResidualViscCoeffs import computeResidualViscCoeffs

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
       
       OPS = sol0.shape[0]
       
       def computeRHSUpdate(fields, U, Dynamics, DynSGS, Rayleigh):
              if Dynamics:
                     rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DDXM_GML, DDZM_GML, DZDX, RdT_bar, fields, U)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendencies at essential boundary DOF
                     rhs[zeroDex[0],0] *= 0.0
                     rhs[zeroDex[1],1] *= 0.0
                     rhs[zeroDex[2],2] *= 0.0
                     rhs[zeroDex[3],3] *= 0.0
              if DynSGS:
                     rhs = tendency.computeDynSGSTendency(RESCF, DDXM, DDZM, DZDX, fields, udex, wdex, pdex, tdex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendency at all boundary DOF
                     rhs[extDex[0],0] *= 0.0
                     rhs[extDex[1],1] *= 0.0
                     rhs[extDex[2],2] *= 0.0
                     rhs[extDex[3],3] *= 0.0
              if Rayleigh:
                     rhs = tendency.computeRayleighTendency(REFG, fields) 
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
       
       def ssprk22(sol, Dynamics, DynSGS, Rayleigh):
              # Stage 1
              U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
              rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
              
              sol1 = computeUpdate(1.0, sol, rhs)
              
              # Stage 2
              U = tendency.computeWeightFields(REFS, sol1, INIT, udex, wdex, pdex, tdex)
              rhs = computeRHSUpdate(sol1, U, Dynamics, DynSGS, Rayleigh)
              
              sol = computeUpdate(0.5, sol1, rhs)
              
              sol = np.array(0.5 * (sol + sol1))
              
              return sol, rhs
       
       def ssprk34(sol, Dynamics, DynSGS, Rayleigh):
              # Stage 1
              U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
              rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
              
              sol1 = computeUpdate(0.5, sol, rhs)
              # Stage 2
              U = tendency.computeWeightFields(REFS, sol1, INIT, udex, wdex, pdex, tdex)
              rhs = computeRHSUpdate(sol1, U, Dynamics, DynSGS, Rayleigh)
              
              sol2 = computeUpdate(0.5, sol1, rhs)
              # Stage 3
              sol = np.array(2.0/3.0 * sol + 1.0 / 3.0 * sol2)
              U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
              rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
              
              sol1 = computeUpdate(1.0/6.0, sol, rhs)
              # Stage 4
              U = tendency.computeWeightFields(REFS, sol1, INIT, udex, wdex, pdex, tdex)
              rhs = computeRHSUpdate(sol1, U, Dynamics, DynSGS, Rayleigh)
              
              sol = computeUpdate(0.5, sol1, rhs)
              
              return sol, rhs
       
       def ketchenson93(sol, Dynamics, DynSGS, Rayleigh):
              for ii in range(7):
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
                     
                     sol = computeUpdate(c1, sol, rhs)
                     
                     if ii == 1:
                            sol1 = np.array(sol)
                     
              # Compute stage 6 with linear combination
              sol = np.array(c2 * (3.0 * sol1 + 2.0 * sol))
              
              # Compute stages 7 - 9 (diffusion applied here)
              for ii in range(2):
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
                     
                     sol = computeUpdate(c1, sol, rhs)
                     
              return sol, rhs
       
       def ketchenson104(sol, Dynamics, DynSGS, Rayleigh):
              sol1 = np.array(sol)
              for ii in range(1,6):
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
                     
                     sol = computeUpdate(c1, sol, rhs)
              
              sol1 = np.array(0.04 * sol1 + 0.36 * sol)
              sol = np.array(15.0 * sol1 - 5.0 * sol)
              
              for ii in range(6,10):
                     U = tendency.computeWeightFields(REFS, sol, INIT, udex, wdex, pdex, tdex)
                     rhs = computeRHSUpdate(sol, U, Dynamics, DynSGS, Rayleigh)
                     
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
       
       # Compute diffusion update
       if DynSGS:
              QM = bn.nanmax(sol, axis=0)
              RES = 1.0 / DT * (sol - sol0) + rhsDyn
              RESCF = computeResidualViscCoeffs(RES, QM, DX, DZ)
              # Use the locally defined custom methods...
              sol, rhsSGS = ssprk34(sol, False, True, False)
              
       # Compute the Rayleigh update outside loop
       #sol, rhsRL = ssprk22(sol, False, False, True)
              
       return sol, (rhsDyn + rhsSGS)