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
       
       def rhsEval(time, fields):
              rhs = tendency.computeDiffusiveFluxTendency(RESCF, DDXM, DDZM, DZDX, fields)
              rhs[extDex,:] *= 0.0
              
              return rhs
       
       def computeRHSUpdate(fields, Dynamics, DynSGS, FlowDiff2):
              if Dynamics:
                     # Scale background wind and shear with ramp up factor
                     U = fields[:,0] + uRamp * INIT[udex]
                     (REFG[4])[:,1] *= uRamp
                     # Compute dynamical tendencies
                     rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFG, DDXM_GML, DDZM_GML, DZDX, RdT_bar, fields, U)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     #rhs[zeroDex[0],0] *= 0.0
                     #rhs[zeroDex[1],1] *= 0.0
                     #rhs[zeroDex[2],2] *= 0.0
                     #rhs[zeroDex[3],3] *= 0.0
              if DynSGS:
                     rhs = tendency.computeDiffusiveFluxTendency(RESCF, DDXM, DDZM, DZDX, fields, extDex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendency at all boundary DOF
                     #rhs[extDex,:] *= 0.0
              if FlowDiff2:
                     rhs = tendency.computeDiffusionTendency(RESCF, DDXM, DDZM, DZDX, fields, extDex)
                     rhs += tendency.computeRayleighTendency(REFG, fields)
                     # Null tendency at all boundary DOF
                     #rhs[extDex,:] *= 0.0
                     
              rhs[zeroDex[0],0] *= 0.0
              rhs[zeroDex[1],1] *= 0.0
              rhs[zeroDex[2],2] *= 0.0
              rhs[zeroDex[3],3] *= 0.0
                     
              return rhs
       
       def computeUpdate(coeff, solA, rhs):
              #Apply updates
              dsol = coeff * DT * rhs
              solB = solA + dsol
              U = solB[:,0] + (uRamp * INIT[udex])
              solB[botDex,1] = np.array(dHdX * U[botDex])
              
              return solB
       
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
       
       def dopri54(sol, Dynamics, DynSGS, FlowDiff):
              rhs1 = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              # Stage 1
              sol1 = computeUpdate(0.2, sol, rhs1)
              # Stage 2
              rhs2 = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              del(sol1)
              sol2 = computeUpdate(3.0 / 40.0, sol, rhs1)
              sol2 = computeUpdate(9.0 / 40.0, sol2, rhs2)
              # Stage 3
              rhs3 = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              del(sol2)
              sol3 = computeUpdate(44.0 / 45.0, sol, rhs1)
              sol3 = computeUpdate(-56.0 / 15.0, sol3, rhs2)
              sol3 = computeUpdate(32.0 / 9.0, sol3, rhs3)
              # Stage 4
              rhs4 = computeRHSUpdate(sol3, Dynamics, DynSGS, FlowDiff)
              del(sol3)
              sol4 = computeUpdate(19372.0 / 6561.0, sol, rhs1)
              sol4 = computeUpdate(-25360.0 / 2187.0, sol4, rhs2)
              sol4 = computeUpdate(64448.0 / 6561.0, sol4, rhs3)
              sol4 = computeUpdate(-212.0 / 729.0, sol4, rhs4)
              # Stage 5
              rhs5 = computeRHSUpdate(sol4, Dynamics, DynSGS, FlowDiff)
              del(sol4)
              sol5 = computeUpdate(-9017.0 / 3168.0, sol, rhs1)
              sol5 = computeUpdate(-355.0 / 33.0, sol5, rhs2)
              sol5 = computeUpdate(46732.0 / 5247.0, sol5, rhs3)
              sol5 = computeUpdate(49.0 / 176.0, sol5, rhs4)
              sol5 = computeUpdate(-5103.0 / 18656.0, sol5, rhs5)
              # Stage 6
              rhs6 = computeRHSUpdate(sol5, Dynamics, DynSGS, FlowDiff)
              del(sol5)
              sol6 = computeUpdate(35.0 / 384.0, sol, rhs1)
              sol6 = computeUpdate(500.0 / 1113.0, sol6, rhs3)
              sol6 = computeUpdate(125.0 / 192.0, sol6, rhs4)
              sol6 = computeUpdate(-2187.0 / 6784.0, sol6, rhs5)
              sol6 = computeUpdate(11.0 / 84.0, sol6, rhs6)
              
              return sol6, rhs6
       
       def kutta384(sol, Dynamics, DynSGS, FlowDiff):
              rhs1 = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              # Stage 1
              sol1 = computeUpdate(1.0 / 3.0, sol, rhs1)
              # Stage 2
              rhs2 = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              del(sol1)
              sol2 = computeUpdate(-1.0 / 3.0, sol, rhs1)
              sol2 = computeUpdate(1.0, sol2, rhs2)
              # Stage 3
              rhs3 = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              del(sol2)
              sol3 = computeUpdate(1.0, sol, rhs1)
              sol3 = computeUpdate(-1.0, sol3, rhs2)
              sol3 = computeUpdate(1.0, sol3, rhs3)
              # Stage 4
              rhs4 = computeRHSUpdate(sol3, Dynamics, DynSGS, FlowDiff)
              del(sol3)
              sol4 = computeUpdate(0.125, sol, rhs1)
              sol4 = computeUpdate(0.375, sol4, rhs2)
              sol4 = computeUpdate(0.375, sol4, rhs3)
              sol4 = computeUpdate(0.125, sol4, rhs4)
              
              return sol4, rhs4
       
       def generic3(sol, Dynamics, DynSGS, FlowDiff):
              alpha = 0.5
              
              rhs1 = computeRHSUpdate(sol, Dynamics, DynSGS, FlowDiff)
              # Stage 1
              sol1 = computeUpdate(alpha, sol, rhs1)
              # Stage 2
              a1 = 1.0 + (1.0 - alpha) / (alpha * (3.0 * alpha - 2.0))
              a2 = - (1.0 - alpha) / (alpha * (3.0 * alpha - 2.0))
              rhs2 = computeRHSUpdate(sol1, Dynamics, DynSGS, FlowDiff)
              del(sol1)
              sol2 = computeUpdate(a1, sol, rhs1)
              sol2 = computeUpdate(a2, sol2, rhs2)
              # Stage 3
              b1 = 0.5 - 1.0 / (6.0 * alpha)
              b2 = 1.0 / (6.0 * alpha * (1.0 - alpha))
              b3 = (2.0 - 3.0 * alpha) / (6.0 * (1.0 -alpha))
              rhs3 = computeRHSUpdate(sol2, Dynamics, DynSGS, FlowDiff)
              del(sol2)
              sol3 = computeUpdate(b1, sol, rhs1)
              sol3 = computeUpdate(b2, sol3, rhs2)
              sol3 = computeUpdate(b3, sol3, rhs3)
              
              return sol3, rhs3
       
       def ketchenson93(sol, Dynamics, DynSGS, FlowDiff):
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
              solf, rhsDyn = ketchenson93(sol0, True, False, False)
       elif order == 4:
              solf, rhsDyn = ketchenson104(sol0, True, False, False)
              
       # Get advective flow velocity components
       U = np.abs(solf[:,0] + uRamp * INIT[udex])
       W = np.abs(solf[:,1])
       if DynSGS:
              # DynSGS compute coefficients
              QM = bn.nanmax(np.abs(solf), axis=0)
              # Estimate the residual
              RES = computeRHSUpdate(solf, True, False, False)
              # Compute diffusion coefficients
              RESCF = dcoeffs.computeResidualViscCoeffs(RES, QM, U, W, DX, DZ)
              
              # Compute a step of diffusion
              solf, rhsDiff = generic3(solf, False, True, False)
              # Compute an adaptive step of diffusion
              '''
              from scipy.integrate import solve_ivp              
              odeSol = solve_ivp(rhsEval, (0.0, DT), solf, method='RK23', t_eval=[0.0, DT], first_step=0.5*DT, max_step=DT, vectorized=True)
              solf = odeSol.y
              print(solf.shape)
              '''
       else: 
              # Flow weighted diffusion (Guerra, 2016)
              RESCF = dcoeffs.computeFlowVelocityCoeffs(U, W, DX, DZ)
              solf, rhsDiff = ssprk34(solf, False, False, True)
       
       # Compute final RHS and return
       rhsOut = computeRHSUpdate(solf, True, False, False)
       return solf, rhsOut