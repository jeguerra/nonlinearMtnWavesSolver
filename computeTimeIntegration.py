#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import math as mt
import scipy.sparse as sps
import time as tm
import bottleneck as bn
import matplotlib.pyplot as plt
import computeEulerEquationsLogPLogT as tendency
import computeResidualViscCoeffs as rescf

def plotRHS(x, rhs, ebcDex, label):
       plim = 2.0E4
       plt.figure(figsize=(10.0, 10.0))
       plt.title(label)
       plt.subplot(2,2,1)
       plt.plot(x, rhs[ebcDex[1],0]); plt.plot(x, rhs[ebcDex[1]+1,0])
       plt.xlim(-plim, plim)
       plt.legend(('level 0', 'level 1'))
       plt.subplot(2,2,2)
       plt.plot(x, rhs[ebcDex[1],1]); plt.plot(x, rhs[ebcDex[1]+1,1])
       plt.xlim(-plim, plim) 
       plt.legend(('level 0', 'level 1'))
       plt.subplot(2,2,3)
       plt.plot(x, rhs[ebcDex[1],2]); plt.plot(x, rhs[ebcDex[1]+1,2])
       plt.xlim(-plim, plim) 
       plt.legend(('level 0', 'level 1'))
       plt.subplot(2,2,4)
       plt.plot(x, rhs[ebcDex[1],3]); plt.plot(x, rhs[ebcDex[1]+1,3])
       plt.xlim(-plim, plim) 
       plt.legend(('level 0', 'level 1'))
       plt.show()
       
       return
       
def computeTimeIntegrationNL(DIMS, PHYS, REFS, REFG, DLD, TOPT, \
                              sol0, rhs0, init0, DCF, zeroDex, ebcDex, \
                              filteredCoeffs, verticalStagger, diffusiveFlux):
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       RLM = REFG[4].data
       dhdx = REFS[6][0]
       bdex = ebcDex[2]
       
       # Use multithreading on CPU
       DDXM_A = REFS[12][0]
       DDZM_A = REFS[12][1]  

       DDXM_B = REFS[13][0]
       DDZM_B = REFS[13][1]
       
       def computeUpdate(coeff, solA, sol2Update, rhsA):
              
              DF = coeff * DT
              
              #%% First dynamics update
              PqPxA, DqDzA = tendency.computeFieldDerivatives(solA, DDXM_A, DDZM_A, verticalStagger)
                                   
              # Compute advection update
              stateA = solA + init0
              rhsAdv = tendency.computeAdvectionLogPLogT_Explicit(PHYS, PqPxA, DqDzA, REFS, REFG, solA, stateA[:,0], stateA[:,1], ebcDex)
              rhsAdv = tendency.enforceTendencyBC(rhsAdv, zeroDex, ebcDex, dhdx)
                     
              # Compute internal force update
              rhsIfc = tendency.computeInternalForceLogPLogT_Explicit(PHYS, PqPxA, DqDzA, REFS, REFG, solA, ebcDex)
              rhsIfc = tendency.enforceTendencyBC(rhsIfc, zeroDex, ebcDex, dhdx)

              # Store the dynamic RHS
              rhsDyn = (rhsAdv + rhsIfc)
              
              #%% Compute diffusive update
              if diffusiveFlux:
                     PqPxA *= DCF[0]
                     DqDzA *= DCF[1]
                            
              # Compute diffusive tendency
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(PqPxA, DqDzA, DDXM_B, DDZM_B, REFS)
              
              rhsDif = tendency.computeDiffusionTendency(solA, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, DLD, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply update
              solB = sol2Update + DF * (rhsDyn + rhsDif)
              
              # Apply Rayleigh damping layer implicitly
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM)
              rdex = REFG[-1]
              solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
              
              return solB, rhsDyn
       
       def ketchesonM2(sol):
              m = 5
              c1 = 1 / (m-1)
              c2 = 1 / m
              sol1 = np.copy(sol)
              for ii in range(m):
                     if ii == m-1:
                            sol1 = c2 * ((m-1) * sol + sol1)
                            sol = computeUpdate(c2, sol, sol1)
                     else:
                            sol = computeUpdate(c1, sol, sol)
                      
              return sol
       
       def ssprk32(sol):
              rhs = np.copy(rhs0)
              # Stage 1
              sol1, rhs, res = computeUpdate(0.5, sol0, sol0, rhs)
              # Stage 2
              sol2, rhs, res = computeUpdate(0.5, sol1, sol1, rhs)
              
              # Stage 3
              sol3 = 1.0 / 3.0 * sol + 2.0 / 3.0 * sol2
              sol, rhs, res = computeUpdate(1.0 / 3.0, sol2, sol3, rhs)

              return sol, rhs, res#, dcf
       
       def ssprk43(sol):
              rhs = 0.0
              # Stage 1
              sol1, rhs = computeUpdate(0.5, sol, sol, rhs)
              rhs += rhs
              # Stage 2
              sol2, rhs = computeUpdate(0.5, sol1, sol1, rhs)
              rhs += rhs
              
              # Stage 3 from SSPRK32
              sols = 1.0 / 3.0 * sol + 2.0 / 3.0 * sol2
              sol3, rhs = computeUpdate(1.0 / 3.0, sol2, sols, rhs)
              rhs += rhs
              
              sols = 0.5 * (sol3 + sol)
              # Stage 3
              #sols, rhs, res = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              #sol3, rhs, res = computeUpdate(1.0 / 6.0, sol2, sols, rhs)
              
              # Stage 4
              sol4, rhs = computeUpdate(0.5, sols, sols, rhs)
              rhs += rhs
              
              res = sol4 - sol3
              rhsAvg = 0.25 * rhs
                                          
              return sol4, rhs
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol, rhs = computeUpdate(c1, sol, sol, 0.0)
              rhs += rhs
              sol1 = np.copy(sol)
              
              for ii in range(5):
                     sol, rhs = computeUpdate(c1, sol, sol, rhs)
                     rhs += rhs
                     
              # Compute stage 6* with linear combination
              sol = 0.6 * sol1 + 0.4 * sol
              
              for ii in range(3):
                     sol, rhs = computeUpdate(c1, sol, sol, rhs)
                     rhs += rhs
                     
              rhsAvg = 1.0/9.0 * rhs
                            
              return sol, rhsAvg
       
       def ketcheson104(sol):
              rhs = 0.0
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol1 = np.copy(sol)
              sol2 = np.copy(sol)
              for ii in range(4):
                     sol1, rhs = computeUpdate(c1, sol1, sol1, rhs)
                     rhs += rhs
              
              sol2 = 0.04 * sol2 + 0.36 * sol1
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1, rhs = computeUpdate(c1, sol1, sol1, rhs)
                     rhs += rhs
                     
              sol = sol2 + 0.6 * sol1
              sol, rhs = computeUpdate(0.1, sol1, sol, rhs)
              rhs += rhs
                            
              rhsAvg = 1.0/9.0 * rhs
              
              return sol, rhsAvg
       
       def ssprk54(sol):
              
              rhs = 0.0
              
              # Stage 1
              b10 = 0.391752226571890
              sol1, rhs = computeUpdate(b10, sol, sol, rhs)
              rhs += rhs
              
              # Stage 2
              a0 = 0.444370493651235
              a1 = 0.555629506348765
              sols = a0 * sol + a1 * sol1
              b21 = 0.368410593050371
              sol2, rhs = computeUpdate(b21, sol1, sols, rhs)
              rhs += rhs
              
              # Stage 3
              a0 = 0.620101851488403
              a2 = 0.379898148511597
              sols = a0 * sol + a2 * sol2
              b32 = 0.251891774271694
              sol3, rhs = computeUpdate(b32, sol2, sols, rhs)
              rhs += rhs
              
              # Stage 4
              a40 = 0.178079954393132
              a43 = 0.821920045606868
              sols = a40 * sol + a43 * sol3
              b43 = 0.544974750228521
              sol4, rhs = computeUpdate(b43, sol3, sols, rhs)
              fun3 = (sol4 - sols) / b43
              rhs += rhs
              
              # Stage 5
              a52 = 0.517231671970585
              a53 = 0.096059710526147
              a54 = 0.386708617503269
              b53 = 0.063692468666290
              b54 = 0.226007483236906
              sols = a52 * sol2 + a53 * sol3 + a54 * sol4
              funs = DT * (b53 * fun3)
              sol5, rhs = computeUpdate(b54, sol4, sols + funs, rhs)
              rhs += rhs
              
              rhsAvg = 0.2 * rhs
              
              return sol5, rhsAvg

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketchesonM2(sol0)
       elif order == 3:
              solB, rhsAvg = ketcheson93(sol0)
       elif order == 4:
              solB, rhsAvg = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketchesonM2(sol0)
       
       state = solB + init0
       # Compute the residual using the average RHS over the time increment
       res = rhsAvg - rhs0
       dcf = rescf.computeResidualViscCoeffs(DIMS, state, rhsAvg, res, DLD, bdex, filteredCoeffs)
       
       return solB, rhsAvg, res, dcf