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
              
              # Apply update
              solB = sol2Update + DF * rhsAdv
                     
              # Compute internal force update
              rhsIfc = tendency.computeInternalForceLogPLogT_Explicit(PHYS, PqPxA, DqDzA, REFS, REFG, solB, ebcDex)
              rhsIfc = tendency.enforceTendencyBC(rhsIfc, zeroDex, ebcDex, dhdx)
              # Apply update
              solB += 1.0 * DF * rhsIfc

              #%% Update the diffusion coefficients
              rhsDyn = (rhsAdv + rhsIfc)
              #resDyn = rhsDyn - rhsA
              #state = solA + init0
              #DCF = rescf.computeResidualViscCoeffs(DIMS, rhsDyn, solA, DLD, bdex, filteredCoeffs)
              
              #%% Compute diffusive update
              #PqPxB, DqDzB = tendency.computeFieldDerivatives(solB, DDXM_A, DDZM_A, verticalStagger)
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
              solB += DF * rhsDif
              
              # Apply Rayleigh damping layer implicitly
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM)
              rdex = REFG[-1]
              solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
              
              return solB, rhsDyn#, resDyn, dcf
       
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
              rhs = np.copy(rhs0)
              # Stage 1
              sol1, rhs, res = computeUpdate(0.5, sol, sol, rhs)
              # Stage 2
              sol2, rhs, res = computeUpdate(0.5, sol1, sol1, rhs)
              
              # Stage 3 from SSPRK32
              sols = 1.0 / 3.0 * sol + 2.0 / 3.0 * sol2
              sol3, rhs, res = computeUpdate(1.0 / 3.0, sol2, sols, rhs)
              
              sols = 0.5 * (sol3 + sol)
              # Stage 3
              #sols, rhs, res = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              #sol3, rhs, res = computeUpdate(1.0 / 6.0, sol2, sols, rhs)
              
              # Stage 4
              sol4, rhs, res = computeUpdate(0.5, sols, sols, rhs)
              
              res = sol4 - sol3
                                          
              return sol4, rhs, res#, dcf
       
       def ssprk53_Opt(sol):
              res = 0.0
              rhs = np.copy(rhs0)
              # Optimized truncation error to SSP coefficient method from Higueras, 2021
              # Stage 1
              c1 = 0.377268915331368
              sol1, rhs, res = computeUpdate(c1, sol, sol, rhs)
              res += res
              
              # Stage 2
              c2 = 0.377268915331368
              sol2, rhs, res = computeUpdate(c2, sol1, sol1, rhs)
              res += res
              
              # Stage 3
              c3 = 0.178557978754048
              sols = np.array(0.526709009150106 * sol + 0.473290990849893 * sol2)
              sol3, rhs, res = computeUpdate(c3, sol2, sols, rhs)
              res += res
              
              # Stage 4
              c4 = 0.321244742913218
              sols = np.array(0.148499306837781 * sol + 0.851500693162219 * sol3)
              sol4, rhs, res = computeUpdate(c4, sol3, sols, rhs)
              res += res
              
              # Stage 5
              sols = np.array(0.166146375373442 * sol1 + 0.063691005483375 * sol2 + 0.770162619143183 * sol4)
              sol5, rhs, res = computeUpdate(0.290558415952914, sol4, sols, rhs)
              res += res
              res *= 0.2
              return sol5, rhs, res
       
       def ketcheson93(sol):
              rhs = np.copy(rhs0); nr = 1.0
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              #c2 = 1.0 / 15.0
              c2 = 1.0 / 42.0
              
              sol0 = np.copy(sol)
              sol, rhs = computeUpdate(c1, sol, sol, rhs)
              rhs += rhs; nr += 1.0
              sol1 = np.copy(sol)
              
              for ii in range(5):
                     sol, rhs = computeUpdate(c1, sol, sol, rhs)
                     rhs += rhs; nr += 1.0
                     
              sols = 1.0/7.0 * sol0 + 6.0/7.0 * sol
              solO2, rhs = computeUpdate(c2, sol, sols, rhs)
              rhs += rhs; nr += 1.0
                     
              # Compute stage 6* with linear combination
              sols = 0.6 * sol1 + 0.4 * sol
              sol = np.copy(sols)
              #sol, rhs, res = computeUpdate(c2, sol, sols, rhs)
              #res += res; nr += 1.0
              
              for ii in range(3):
                     sol, rhs = computeUpdate(c1, sol, sol, rhs)
                     rhs += rhs; nr += 1.0
                     
              resErr = sol - solO2
              rhsAvg = 1.0/nr * rhs
                            
              return sol, rhsAvg, resErr
       
       def ketcheson104(sol):
              rhs = np.copy(rhs0); nr = 1.0
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol1 = np.copy(sol)
              sol2 = np.copy(sol)
              for ii in range(4):
                     sol1, rhs = computeUpdate(c1, sol1, sol1, rhs)
                     rhs += rhs; nr += 1.0
              
              sol2 = 0.04 * sol2 + 0.36 * sol1
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1, rhs = computeUpdate(c1, sol1, sol1, rhs)
                     rhs += rhs; nr += 1.0
                     
              sol = sol2 + 0.6 * sol1
              sol, rhs = computeUpdate(0.1, sol1, sol, rhs)
              rhs += rhs; nr += 1.0
                            
              rhsAvg = 1.0 / nr * rhs
              
              return sol, rhsAvg
       
       def ssprk54(sol, DTF):
              
              rhs = 0.0
              
              # Stage 1
              b10 = 0.391752226571890
              sol1, rhs, res = computeUpdate(DTF * b10, sol, sol, rhs)
              
              # Stage 2
              a0 = 0.444370493651235
              a1 = 0.555629506348765
              sols = a0 * sol + a1 * sol1
              b21 = 0.368410593050371
              sol2, rhs, res = computeUpdate(DTF * b21, sol1, sols, rhs)
              
              # Stage 3
              a0 = 0.620101851488403
              a2 = 0.379898148511597
              sols = a0 * sol + a2 * sol2
              b32 = 0.251891774271694
              sol3, rhs, res = computeUpdate(DTF * b32, sol2, sols, rhs)
              fun3 = (sol3 - sols) / (DTF * b32)
              
              # Stage 4
              a40 = 0.178079954393132
              a43 = 0.821920045606868
              sols = a40 * sol + a43 * sol3
              b43 = 0.544974750228521
              sol4, rhs, res = computeUpdate(DTF * b43, sol3, sols, rhs)
              
              # Stage 5
              a52 = 0.517231671970585
              a53 = 0.096059710526147
              a54 = 0.386708617503269
              b53 = 0.063692468666290
              b54 = 0.226007483236906
              sols = a52 * sol2 + a53 * sol3 + a54 * sol4
              funs = (DTF * DT) * (b53 * fun3)
              sol5, rhs, res = computeUpdate(b54, sol4, sols + funs, rhs)
              
              return sol5, rhs, res#, dcf

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketchesonM2(sol0)
       elif order == 3:
              solB, rhs, err = ketcheson93(sol0)
       elif order == 4:
              solB, rhs = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketchesonM2(sol0)
       
       #state = solB + init0
       res = (solB - sol0) / DT - rhs
       dcf = rescf.computeResidualViscCoeffs(DIMS, res, solB, DLD, bdex, filteredCoeffs)
       
       return solB, rhs, res, dcf