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

# Computes an update of the dynamics only
def computeTimeIntegrationNL1(DIMS, PHYS, REFS, REFG, DLD, TOPT, \
                              sol0, init0, zeroDex, ebcDex, \
                              isFirstStep, filteredCoeffs, \
                              verticalStagger, DynSGS_RES, NE):
       DT = TOPT[0]
       mu = REFG[3]
       RLM = REFG[4].data
       ldex = REFG[5]
       dhdx = REFS[6][0]       
       diffusiveFlux = False
       
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
       
       def computeUpdate1(coeff, solA, sol2Update):
              
              DF = coeff * DT
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM)
              
              # Compute dynamics RHS
              args = [solA, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False]
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
                                       
              return solB, rhsExp
       
       def computeUpdate2(coeff, solA, sol2Update, DCF):
              
              DF = coeff * DT
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM)
              
              # Compute first derivatives
              if verticalStagger:
                     DqDxA, DqDzA = tendency.computeFieldDerivativeStag(solA, DDXM_A, DDZM_A)
              else:
                     DqDxA, DqDzA = tendency.computeFieldDerivatives(solA, DDXM_A, DDZM_A)
              
              # Compute terrain following adjustments
              DqDxA -= REFS[15] * DqDzA
              
              # Compute diffusive fluxes
              if diffusiveFlux:
                     DqDxA *= np.expand_dims(DCF[0], axis=1)
                     DqDzA *= np.expand_dims(DCF[0], axis=1)
              
              #'''
              # Compute second derivatives
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(DqDxA, DqDzA, DDXM_B, DDZM_B, REFS, REFG, DCF)
              
              # Compute diffusive tendency
              rhsDif = tendency.computeDiffusionTendency(solA, DqDxA, DqDzA, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, DLD, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply diffusion update
              try:
                     solB = sol2Update + DF * rhsDif
              except FloatingPointError:
                     solB = np.copy(sol2Update)
                     
              # Apply Rayleigh damping layer implicitly
              rdex = [0, 1, 2, 3]
              solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
              
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              
              return solB
       
       def ketcheson93(sol):
              
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              c2 = 1.0 / 15.0
              
              #sol0 = np.copy(sol)
              sol1, rhs0 = computeUpdate1(c1, sol, sol)
              sol2 = np.copy(sol1)
              
              for ii in range(4):
                     sol1, rhs = computeUpdate1(c1, sol1, sol1)
              
              # Compute stage 6
              sol2 = 0.6 * sol2 + 0.4 * sol1
              sol1, rhs = computeUpdate1(c2, sol1, sol2)
              
              for ii in range(3):
                     sol1, rhs = computeUpdate1(c1, sol1, sol1)
                                                 
              return sol1, rhs0
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol1 = np.copy(sol)
              sol2 = np.copy(sol)
              for ii in range(4):
                     if ii == 0:
                            sol1, rhs0 = computeUpdate1(c1, sol1, sol1)
                     else:
                            sol1, rhs = computeUpdate1(c1, sol1, sol1)
              
              sol2 = 0.04 * sol2 + 0.36 * sol1
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1, rhs = computeUpdate1(c1, sol1, sol1)
                     
              sol = sol2 + 0.6 * sol1
              sol, rhs = computeUpdate1(0.1, sol1, sol)
              
              return sol, rhs0
       
       def ssprk54(sol, DTF):
              
              # Stage 1
              b1 = 0.39175222700392
              sol1, rhs0 = computeUpdate1(DTF * b1, sol, sol)
              
              # Stage 2
              sols = 0.44437049406734 * sol + 0.55562950593266 * sol1
              b2 = 0.36841059262959
              sol2, rhs = computeUpdate1(DTF * b2, sol1, sols)
              
              # Stage 3
              sols = 0.62010185138540 * sol + 0.37989814861460 * sol2
              b3 = 0.25189177424738
              sol3, rhs = computeUpdate1(DTF * b3, sol2, sols)
              
              # Stage 4
              a40 = 0.17807995410773
              a43 = 0.82192004589227
              sols = a40 * sol + a43 * sol3
              b43 = 0.54497475021237
              sol4, rhs = computeUpdate1(DTF * b43, sol3, sols)
              
              # Stage 5
              a50 = 0.00683325884039
              a52 = 0.51723167208978
              a53 = 0.12759831133288
              a54 = 0.34833675773694
              b53 = 0.08460416338212
              b54 = 0.22600748319395
              sols = (a50 - a40 * b53 / b43) * sol + a52 * sol2 + \
                     (a53 - a43 * b53 / b43) * sol3 + (a54 + b53 / b43) * sol4
              sol5, rhs = computeUpdate1(DTF * b54, sol4, sols)
              
              return sol5, rhs0
       
       def ketchesonM2(sol, DCF):
              m = 5
              c1 = 1 / (m-1)
              c2 = 1 / m
              sol1 = np.copy(sol)
              for ii in range(m):
                     if ii == m-1:
                            sol1 = c2 * ((m-1) * sol + sol1)
                            sol = computeUpdate2(c2, sol, sol1, DCF)
                     elif ii == 0:
                            sol = computeUpdate2(c1, sol, sol, DCF)
                     else:
                            sol = computeUpdate2(c1, sol, sol, DCF)
                      
              return sol
       
       def ssprk32(sol, DCF, DTF):
              # Stage 1
              sol1 = computeUpdate2(0.5 * DTF, sol, sol, DCF)
              # Stage 2
              sol2 = computeUpdate2(0.5 * DTF, sol1, sol1, DCF)
              
              # Stage 3
              sol3 = np.array(1.0 / 3.0 * sol + 2.0 / 3.0 * sol2)
              sol = computeUpdate2(1.0 / 3.0 * DTF, sol2, sol3, DCF)

              return sol
       
       def ssprk33(sol, DCF, DTF):
              # Stage 1
              sol1 = computeUpdate2(1.0 * DTF, sol, sol, DCF)
              
              # Stage 2
              sols = np.array(0.75 * sol + 0.25 * sol1)
              sol2 = computeUpdate2(0.25 * DTF, sol1, sols, DCF)
              
              # Stage 3
              sols = np.array(1.0 / 3.0 * sol + 2.0 / 3.0 * sol2)
              sol3 = computeUpdate2(2.0 / 3.0 * DTF, sol2, sols, DCF)
                                          
              return sol3
       
       def ssprk43(sol, DCF, DTF):
              # Stage 1
              sol1 = computeUpdate2(0.5 * DTF, sol, sol, DCF)
              # Stage 2
              sol2 = computeUpdate2(0.5 * DTF, sol1, sol1, DCF)
              
              # Stage 3
              sols = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              sol3 = computeUpdate2(1.0 / 6.0 * DTF, sol2, sols, DCF)
              
              # Stage 4
              sol4 = computeUpdate2(0.5 * DTF, sol3, sol3, DCF)
                                          
              return sol4
       
       def ssprk53_Opt(sol, DCF):
              
              # Optimized truncation error to SSP coefficient method from Higueras, 2021
              # Stage 1
              c1 = 0.377268915331368
              sol1 = computeUpdate2(c1, sol, sol, DCF)
              
              # Stage 2
              c2 = 0.377268915331368
              sol2 = computeUpdate2(c2, sol1, sol1, DCF)
              
              # Stage 3
              c3 = 0.178557978754048
              sols = np.array(0.526709009150106 * sol + 0.473290990849893 * sol2)
              sol3 = computeUpdate2(c3, sol2, sols, DCF)
              
              # Stage 4
              c4 = 0.321244742913218
              sols = np.array(0.148499306837781 * sol + 0.851500693162219 * sol3)
              sol4 = computeUpdate2(c4, sol3, sols, DCF)
              
              # Stage 5
              sols = np.array(0.166146375373442 * sol1 + 0.063691005483375 * sol2 + 0.770162619143183 * sol4)
              sol5 = computeUpdate2(0.290558415952914, sol4, sols, DCF)
              
              return sol5
       
       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       #solB, rhsOld = ketcheson93(sol0)
       #solB, rhsOld = ketcheson104(sol0)
       
       # Solve to first half step
       solA, rhsOld = ssprk54(sol0, 0.5)
       # Solve to last half step
       solB, rhsMid = ssprk54(solA, 0.5)
              
       # Update the adaptive coefficients using residual or right hand side
       #args = [solB, init0, DDXM_B, DDZM_B, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, False, True]
       #rhsNew, DqDxA, DqDzA = tendency.computeRHS(*args)
       #args = [solB, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False]
       #rhsNew, DqDxA, DqDzA = tendency.computeRHS(*args)
       
       # Normalization and bounding to DynSGS
       state = solB + init0
       qnorm = (solB - bn.nanmean(solB))
       #'''
       if DynSGS_RES:
              resField = 2.0 * (solB - solA) / DT - 0.5 * (rhsMid + rhsOld)
       else:
              resField = 0.5 * (rhsMid + rhsOld)
       resField *= 2.0
       #'''
       if filteredCoeffs:
              DCF = rescf.computeResidualViscCoeffsFiltered(DIMS, resField, qnorm, state, DLD, NE)
       else:
              DCF = rescf.computeResidualViscCoeffsRaw(DIMS, resField, qnorm, state, DLD, dhdx, ebcDex[2], ldex)
       #del(resField)       
       
       # Compute the diffusion update
       #solB = ssprk43(solB, DCF)
       #solB = ssprk53_Opt(solB, DCF)
       
       #solB = ketchesonM2(solB, DCF)
       
       solB = ssprk43(solB, DCF, 0.5)
       solB = ssprk33(solB, DCF, 0.5)

       #solB = ssprk32(solB, DCF, 0.5)
       #solB = ssprk32(solB, DCF, 0.5)
       
       return solB
       
def computeTimeIntegrationNL2(DIMS, PHYS, REFS, REFG, DLD, TOPT, \
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
       #state = sol0 + init0
       #qnorm = (sol0 - bn.nanmean(sol0))
       
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
              
              DF = coeff * DT
              
              # Compute dynamics RHS
              args = [solA, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False]
              rhsExp, DqDxA, DqDzA = tendency.computeRHS(*args)
              
              try:
                  solB = sol2Update + DF * rhsExp
              except FloatingPointError:
                  solB = sol2Update + 0.0
                  
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              
              # Update the adaptive coefficients using residual or right hand side
              args = [solB, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False]
              rhsNew, DqDxB, DqDzB = tendency.computeRHS(*args)
              
              # Normalization and bounding to DynSGS
              state = solB + init0
              qnorm = (solB - bn.nanmean(solB))
              
              if DynSGS_RES:
                     resField = rhsExp - rhsNew
              else:
                     resField = 0.5 * (rhsExp + rhsNew)
              resField *= 2.0
       
              if filteredCoeffs:
                     DCF = rescf.computeResidualViscCoeffsFiltered(DIMS, resField, qnorm, state, DLD, NE)
              else:
                     DCF = rescf.computeResidualViscCoeffsRaw(DIMS, resField, qnorm, state, DLD, dhdx, ebcDex[2], ldex)
              del(resField)
              
              # Compute diffusive fluxes
              if diffusiveFlux:
                     DqDxB *= np.expand_dims(DCF[0], axis=1)
                     DqDzB *= np.expand_dims(DCF[0], axis=1)
              
              # Compute diffusive tendency
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(DqDxB, DqDzB, DDXM_B, DDZM_B, REFS, REFG, DCF)
              
              rhsDif = tendency.computeDiffusionTendency(solB, DqDxB, DqDzB, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
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
              '''
              # Apply Rayleigh damping layer implicitly
              RayDamp = np.reciprocal(1.0 + DF * mu * RLM)
              rdex = [0, 1, 2, 3]
              solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
              
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              '''
              return solB
       
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
              
              sol1 = np.copy(sol)
              sol2 = np.copy(sol)
              for ii in range(4):
                     if ii == 0:
                            sol1 = computeUpdate(c1, sol1, sol1)
                     else:
                            sol1 = computeUpdate(c1, sol1, sol1)
              
              sol2 = 0.04 * sol2 + 0.36 * sol1
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1 = computeUpdate(c1, sol1, sol1)
                     
              sol = sol2 + 0.6 * sol1
              sol = computeUpdate(0.1, sol1, sol)
                            
              return sol

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketchesonM2(sol0)
       elif order == 3:
              solB = ketcheson93(sol0)
       elif order == 4:
              solB = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketchesonM2(sol0)
              
       # Apply Rayleigh damping layer implicitly
       RayDamp = np.reciprocal(1.0 + 2.0 * DT * mu * RLM)
       rdex = [0, 1, 2, 3]
       solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
       
       # Clean up on essential BC
       U = solB[:,0] + init0[:,0]
       solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
       
       return solB