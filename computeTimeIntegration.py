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
                              filteredCoeffs, verticalStagger, \
                              DynSGS_RES):
       DT = TOPT[0]
       mu = REFG[3]
       RLM = REFG[4].data
       ldex = REFG[5]
       dhdx = REFS[6][0]       
       diffusiveFlux = False
       

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
              args = [solA, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False, True]
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
       
       # Solve to first half step
       solA, rhsOld = ssprk54(sol0, 0.5)
       # Solve to last half step
       solB, rhsMid = ssprk54(solA, 0.5)
              
       # Update the adaptive coefficients using residual or right hand side
       args = [solB, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False, True]
       rhsNew, DqDxA, DqDzA = tendency.computeRHS(*args)
       
       # Normalization and bounding to DynSGS
       state = solB + init0
       qnorm = (solB - bn.nanmean(solB))
       #'''
       if DynSGS_RES:
              resField = 2.0 * (solB - solA) / DT - 0.5 * (rhsNew + rhsMid)
       else:
              resField = 0.5 * (rhsNew + rhsMid)
       #'''
       DCF = rescf.computeResidualViscCoeffs(DIMS, resField, qnorm, state, DLD, dhdx, ebcDex[2], filteredCoeffs)
       
       # Solve first half step
       solB = ssprk43(solB, DCF, 0.5)
       # Solve second half step
       solB = ssprk43(solB, DCF, 0.5)
       
       return solB
       
def computeTimeIntegrationNL2(DIMS, PHYS, REFS, REFG, DLD, TOPT, \
                              sol0, init0, DCF, rhs0, zeroDex, ebcDex, \
                              filteredCoeffs, verticalStagger, DynSGS_RES):
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       RLM = REFG[4].data
       dhdx = REFS[6][0]       
       diffusiveFlux = False
       
       # Use multithreading on CPU
       DDXM_A = REFS[12][0]
       DDZM_A = REFS[12][1]  

       DDXM_B = REFS[13][0]
       DDZM_B = REFS[13][1]
                     
       def computeUpdate1(coeff, solA, sol2Update):
              
              DF = coeff * DT
              
              # Compute dynamics RHS
              args = [solA, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, False, verticalStagger, False, True]
              rhsExp, DqDxA, DqDzA = tendency.computeRHS(*args)
              
              # Compute diffusive fluxes
              if diffusiveFlux:
                     DqDxA *= np.expand_dims(DCF[0], axis=1)
                     DqDzA *= np.expand_dims(DCF[1], axis=1)
              
              # Compute diffusive tendency
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(DqDxA, DqDzA, DDXM_B, DDZM_B, REFS, REFG, DCF)
              
              rhsDif = tendency.computeDiffusionTendency(solA, DqDxA, DqDzA, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, DLD, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply update
              try:
                     solB = sol2Update + DF * (rhsExp + rhsDif)
              except FloatingPointError:
                     solB = sol2Update + 0.0
              
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              
              return solB
       
       def computeUpdate(coeff, solA, sol2Update):
              
              DF = coeff * DT
              
              # Get derivatives
              if verticalStagger:
                     PqPx, DqDz = tendency.computeFieldDerivativeStag(solA, DDXM_A, DDZM_A)
              else:
                     PqPx, DqDz = tendency.computeFieldDerivatives(solA, DDXM_A, DDZM_A)
                                   
              #%% Compute advection update
              stateA = solA + init0
              rhsAdv = tendency.computeAdvectionLogPLogT_Explicit(PHYS, PqPx, DqDz, REFS, REFG, solA, stateA[:,0], stateA[:,1], ebcDex)
              rhsAdv = tendency.enforceTendencyBC(rhsAdv, zeroDex, ebcDex, dhdx)
              
              # Apply update
              try:
                     solB = sol2Update + DF * rhsAdv
              except FloatingPointError:
                     solB = np.copy(sol2Update)
                     
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
                     
              #%% Compute internal force update
              rhsIfc = tendency.computeInternalForceLogPLogT_Explicit(PHYS, PqPx, DqDz, REFS, REFG, solB, ebcDex)
              rhsIfc = tendency.enforceTendencyBC(rhsIfc, zeroDex, ebcDex, dhdx)
              
              # Apply update
              try:
                     solB += DF * rhsIfc
              except FloatingPointError:
                     solB += 0.0
                     
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              
              #%% Compute diffusive update
              if diffusiveFlux:
                     PqPx *= DCF[0]
                     DqDz *= DCF[1]
              
              # Compute diffusive tendency
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(PqPx, DqDz, DDXM_B, DDZM_B, REFS, REFG, DCF)
              
              rhsDif = tendency.computeDiffusionTendency(solB, P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, DLD, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply update
              try:
                     solB += DF * rhsDif
              except FloatingPointError:
                     solB += 0.0
              
              # Clean up on essential BC
              U = solB[:,0] + init0[:,0]
              solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
              
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
              try:
                     sol2 *= 0.04
              except FloatingPointError:
                     sol2 *= 0.0
              try:
                     sol2 += 0.36 * sol1
              except FloatingPointError:
                     sol2 += 0.0
              
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1 = computeUpdate(c1, sol1, sol1)
              try:       
                     sol = sol2 + 0.6 * sol1
              except FloatingPointError:
                     sol = sol2 + 0.0
                     
              sol = computeUpdate(0.1, sol1, sol)
                            
              return sol
       
       def ssprk84(sol):
              
              # Stage 1
              b0 = 0.24120020561311
              sol1 = computeUpdate(b0, sol, sol)
              
              # Stage 2
              sols = 0.10645325745007 * sol + 0.89354674254993 * sol1
              b1 = 0.21552365802797
              sol2 = computeUpdate(b1, sol1, sols)
              
              # Stage 3
              b2 = 0.24120020561311
              sol3 = computeUpdate(b2, sol2, sol2)
              fun2 = (sol3 - sol2) / (b2 * DT)
              
              # Stage 4
              a0 = 0.57175518477257
              a3 = 1.0 - a0
              sols = a0 * sol + a3 * sol3
              b3 = 0.10329273748560
              sol4 = computeUpdate(b3, sol3, sols)
              fun3 = (sol4 - sols) / (b3 * DT)
              
              # Stage 5
              a0 = 0.19161667219044
              a4 = 1.0 - a0
              sols = a0 * sol + a4 * sol4
              b4 = 0.19498222488188
              sol5 = computeUpdate(b4, sol4, sols)
              
              # Stage 6
              b5 = 0.24120020561311
              sol6 = computeUpdate(b5, sol5, sol5)
              fun5 = (sol6 - sol5) / (b5 * DT)
              
              # Stage 7
              b6 = 0.24120020561311
              sol7 = computeUpdate(b6, sol6, sol6)
              
              # Stage 8
              a0 = 0.02580435327923
              a2 = 0.03629901341774
              a3 = 0.31859181340256
              a4 = 0.05186768980103
              a5 = 0.03944076217320
              a6 = 0.00511633747411
              a7 = 0.52288003045213
              b2 = 0.00875532949991
              b3 = 0.06195575835101
              b5 = 0.00951311994571
              b7 = 0.12611877085604
              sols = a0 * sol + a2 * sol2 + a3 * sol3 + a4 * sol4 + a5 * sol5 + a6 * sol6 + a7 * sol7
              funs = DT * (b2 * fun2 + b3 * fun3 + b5 * fun5)
              sol8 = computeUpdate(b7, sol7, sols + funs)
              
              return sol8
       
       def ssprk54(sol, DTF):
              
              # Stage 1
              b10 = 0.39175222700392
              sol1 = computeUpdate(DTF * b10, sol, sol)
              
              # Stage 2
              a0 = 0.44437049406734
              a1 = 0.55562950593266
              sols = a0 * sol + a1 * sol1
              b21 = 0.36841059262959
              sol2 = computeUpdate(DTF * b21, sol1, sols)
              
              # Stage 3
              a0 = 0.62010185138540
              a2 = 0.37989814861460
              sols = a0 * sol + a2 * sol2
              b32 = 0.25189177424738
              sol3 = computeUpdate(DTF * b32, sol2, sols)
              fun3 = (sol3 - sols) / (DTF * b32)
              
              # Stage 4
              a40 = 0.17807995410773
              a43 = 0.82192004589227
              sols = a40 * sol + a43 * sol3
              b43 = 0.54497475021237
              sol4 = computeUpdate(DTF * b43, sol3, sols)
              fun4 = (sol4 - sols) / (DTF * b43)
              
              # Stage 5
              a50 = 0.00683325884039
              a52 = 0.51723167208978
              a53 = 0.12759831133288
              a54 = 0.34833675773694
              b53 = 0.08460416338212
              b54 = 0.22600748319395
              sols = a50 * sol + a52 * sol2 + a53 * sol3 + a54 * sol4
              funs = (DTF * DT) * (b53 * fun3 + b54 * fun4)
              sol5 = computeUpdate(b54, sol4, sols + funs)
              
              return sol5

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketchesonM2(sol0)
       elif order == 3:
              solB = ketcheson93(sol0)
       elif order == 4:
              #solB = ssprk54(sol0, 1.0)
              #solB = ssprk84(sol0)
              solB = ketcheson104(sol0)
              '''
              solA = ssprk54(sol0, 0.5)
              
              # Apply Rayleigh damping layer implicitly
              RayDamp = np.reciprocal(1.0 + 1.0 * DT * mu * RLM)
              rdex = [0, 1, 2, 3]
              solA[:,rdex] = np.copy(RayDamp.T * solA[:,rdex])
              
              # Clean up on essential BC
              U = solA[:,0] + init0[:,0]
              solA = tendency.enforceEssentialBC(solA, U, zeroDex, ebcDex, dhdx)
              
              args = [solA, init0, DDXM_A, DDZM_A, dhdx, PHYS, REFS, REFG, ebcDex, zeroDex, True, verticalStagger, True, True]
              rhsMid, DqDxA, DqDzA = tendency.computeRHS(*args)
              
              # Normalization and bounding to DynSGS
              state = solA + init0
              qnorm = (solA - bn.nanmean(solA))
              
              if DynSGS_RES:
                     resField = (solA - sol0) / (0.5 * DT) - 0.5 * (rhsMid + rhs0)
              else:
                     resField = 0.5 * (rhsMid + rhs0)
              resField *= 2.0
              
              DCF = rescf.computeResidualViscCoeffsRaw(DIMS, resField, qnorm, state, DLD, dhdx, ebcDex[2], filteredCoeffs)
                     
              solB = ssprk54(solA, 0.5)
              '''
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketchesonM2(sol0)
              
       # Apply Rayleigh damping layer implicitly
       RayDamp = np.reciprocal(1.0 + DT * mu * RLM)
       rdex = REFG[-1]
       solB[:,rdex] = np.copy(RayDamp.T * solB[:,rdex])
       
       # Clean up on essential BC
       U = solB[:,0] + init0[:,0]
       solB = tendency.enforceEssentialBC(solB, U, zeroDex, ebcDex, dhdx)
       
       return solB #, solA, rhsMid