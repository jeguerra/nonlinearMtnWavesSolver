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
                              sol0, init0, zeroDex, ebcDex, \
                              isFirstStep, filteredCoeffs, \
                              verticalStagger, DynSGS_RES, NE):
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       RLM = REFG[4].data
       ldex = REFG[5]
       dhdx = REFS[6][0]       
       diffusiveFlux = False
       
       # Normalization and bounding to DynSGS
       state = sol0 + init0
       qnorm = (sol0 - bn.nanmean(sol0))
       
       #'''
       if isFirstStep:
              # Use SciPY sparse for dynamics
              DDXM_A = REFS[10][0]
              if verticalStagger:
                     DDZM_A = REFS[18]
              else:
                     DDZM_A = REFS[10][1]
       else:
              # Use multithreading on CPU
              DDXM_A = REFS[12][0]
              if verticalStagger:
                     DDZM_A = REFS[19]
              else:
                     DDZM_A = REFS[12][1]  
       
       DDXM_B = REFS[13][0]
       DDZM_B = REFS[13][1]
                     
       def computeUpdate(coeff, solA, sol2Update):
              
              # Normalization and bounding to DynSGS
              #state = solA + init0
              #qnorm = 1.0 * (solA - bn.nanmean(solA))
              
              DF = coeff * DT
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM)
              
              # Compute dynamics RHS
              args = [solA, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger]
              rhsExp, DqDxA, DqDzA = tendency.computeRHS(*args)
              
              try:
                  solB = sol2Update + DF * rhsExp
              except FloatingPointError:
                  solB = sol2Update + 0.0
                  
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)    
                  
              # Apply Rayleigh damping layer implicitly
              rdex = [0, 1, 2, 3]
              solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
              
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              
              # Update the adaptive coefficients using residual
              args = [solB, init0, DDXM_B, DDZM_B, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, False]
              rhsNew, DqDxR, DqDzR = tendency.computeRHS(*args)
              if DynSGS_RES:
                     resField = rhsNew - rhsExp
              else:
                     resField = np.copy(rhsNew)
       
              if filteredCoeffs:
                     DCF = rescf.computeResidualViscCoeffsFiltered(DIMS, resField, qnorm, state, DLD, NE)
              else:
                     DCF = rescf.computeResidualViscCoeffsRaw(DIMS, resField, qnorm, state, DLD, dhdx, ebcDex[2], ldex)
              del(resField)
              
              # Compute diffusive tendency
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(DqDxA, DqDzA, DDXM_B, DDZM_B, REFS, REFG, DCF, diffusiveFlux)
              
              rhsDif = tendency.computeDiffusionTendency(solA, DqDxA, DqDzA, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, DLD, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply diffusion update
              try:
                     solB += DF * rhsDif
              except FloatingPointError:
                     solB += 0.0
              
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
                                          
              return solB
       
       def ssprk43(sol):
              # Stage 1
              sol1 = computeUpdate(0.5, sol, sol)
              # Stage 2
              sol2 = computeUpdate(0.5, sol1, sol1)
              # Stage 3
              sol = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              sol1 = computeUpdate(1.0 / 6.0, sol, sol)
              # Stage 4
              sol = computeUpdate(0.5, sol1, sol1)
              
              return sol
       
       def ssprk53_Opt(sol):
              
              # Optimized truncation error to SSP coefficient method from Higueras, 2021
              # Stage 1
              c1 = 0.377268915331368
              sol1 = computeUpdate(c1, sol, sol)
              
              # Stage 2
              c2 = 0.377268915331368
              sol2 = computeUpdate(c2, sol1, sol1)
              
              # Stage 3
              c3 = 0.178557978754048
              sol3 = np.array(0.526709009150106 * sol + 0.473290990849893 * sol2)
              sol3 = computeUpdate(c3, sol2, sol3)
              
              # Stage 4
              c4 = 0.321244742913218
              sol4 = np.array(0.148499306837781 * sol + 0.851500693162219 * sol3)
              sol4 = computeUpdate(c4, sol3, sol4)
              
              # Stage 5
              sol5 = np.array(0.166146375373442 * sol1 + 0.063691005483375 * sol2 + 0.770162619143183 * sol4)
              sol = computeUpdate(0.290558415952914, sol4, sol5)
              
              return sol
       
       def ketcheson62(sol):
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
       
       def ketcheson93(sol):
              
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 15.0
              
              sol = computeUpdate(c1, sol, sol)
              sol1 = np.copy(sol)
              
              for ii in range(4):
                     sol = computeUpdate(c1, sol, sol)
                     
              # Compute stage 6 with linear combination
              sol1 = np.array(0.6 * sol1 + 0.4 * sol)
              sol = computeUpdate(c2, sol, sol1)
              
              for ii in range(3):
                     sol = computeUpdate(c1, sol, sol)
                     
              return sol
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
       
              sol2 = np.array(sol)
              for ii in range(5):
                     sol = computeUpdate(c1, sol, sol)
              
              sol2 = np.array(0.04 * sol2 + 0.36 * sol)
              sol = np.array(15.0 * sol2 - 5.0 * sol)
              
              for ii in range(4):
                     sol = computeUpdate(c1, sol, sol)
                     
              sol2 = sol2 + 0.6 * sol
              sol = computeUpdate(0.1, sol, sol2)
              
              return sol
       
       # 2 step 8 stage 5th order method (Goolieb, Ketcheson, Shu) 2010
       def tsrk85(y1, y0):
              
              r = 0.447
              
              # Initialize stage evaluations
              dy0 = computeUpdate(1.0/r, y0, y0)
              dy1 = computeUpdate(1.0/r, y1, y1)
              
              # Stage 2
              q20 = 0.085330772948
              q21 = 0.914669227052
              y2 = (1.0 - q20 - q21) * y1 + q20 * dy0 + q21 * dy1
              
              # Stage 3
              q30 = 0.058121281984
              q32 = 0.941878718016
              dy2 = computeUpdate(1.0/r, y2, y2)
              y3 = (1.0 - q30 - q32) * y1 + q30 * dy0 + q32 * dy2
              
              # Stage 4
              q41 = 0.036365639243
              q43 = 0.802870131353
              dy3 = computeUpdate(1.0/r, y3, y3)
              y4 = (1.0 - q41 - q43) * y1 + q41 * dy1 + q43 * dy3
              
              # Stage 5
              q51 = 0.491214340661
              q54 = 0.508785659339
              y5 = (1.0 - q51 - q54) * y1 + q51 * dy1 + q54 * y4
              y5 = computeUpdate(q54/r, y4, y5)
              
              # Stage 6
              q61 = 0.566135231631
              q65 = 0.433864768369
              y6 = (1.0 - q61 - q65) * y1 + q61 * dy1 + q65 * y5
              y6 = computeUpdate(q65/r, y5, y6)
              
              # Stage 7
              d7 = 0.00367418482
              q70 = 0.020705281787
              q71 = 0.091646079652
              q76 = 0.883974453742
              dy6 = computeUpdate(1.0/r, y6, y6)
              y7 = (1.0 - d7 - q70 - q71 - q76) * y1 + d7 * y0 + q70 * dy0 + q71 * dy1 + q76 * dy6
              
              # Stage 8
              q80 = 0.008506650139
              q81 = 0.110261531523
              q82 = 0.030113037742
              q87 = 0.851118780596
              y8 = (1.0 - q80 - q81 - q82 - q87) * y1 + q70 * dy0 + q71 * dy1 + q82 * dy2 + q87 * y7
              y8 = computeUpdate(q87/r, y7, y8)
              
              # Final stage
              n2 = 0.179502832155
              n3 = 0.073789956885
              n6 = 0.017607159013
              n8 = 0.729100051947
              sol = (1.0 - n2 - n3 - n6 - n8) * y1 + n2 * dy2 + n3 * dy3 + n6 * dy6 + n8 * y8
              sol = computeUpdate(n8/r, y8, sol)
              
              return sol

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketcheson62(sol0)
       elif order == 3:
              #solB = ketcheson93(sol0)
              #solB = ssprk43(sol0)
              solB = ssprk53_Opt(sol0)
       elif order == 4:
              solB = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketcheson62(sol0)
       
       return solB