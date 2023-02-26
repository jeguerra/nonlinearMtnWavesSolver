#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import matplotlib.pyplot as plt
import computeEulerEquationsLogPLogT as tendency

# Change floating point errors
np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')

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
                              sol0, init0, DCF, zeroDex, ebcDex, \
                              filteredCoeffs, verticalStagger, diffusiveFlux):
       
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       RLMX = REFG[4][1].data
       RLMZ = REFG[4][2].data
       
       S = DLD[4]
       dhdx = np.expand_dims(REFS[6][0], axis=1)
       bdex = ebcDex[2]
       tdex = ebcDex[3]
       
       # Use multithreading on CPU
       DDXM_A = REFS[12][0]
       DDZM_A = REFS[12][1]  

       DDXM_B = REFS[13][0]
       DDZM_B = REFS[13][1]
       
       def computeUpdate(coeff, solA, sol2Update):
              
              DF = coeff * DT
              
              #%% First dynamics update
              DqDxA, DqDzA = tendency.computeFieldDerivatives(solA, DDXM_A, DDZM_A, verticalStagger)
              PqPxA = DqDxA - REFS[15] * DqDzA
              
              # Apply Rayleigh damping layer implicitly to derivatives
              RayDX = np.reciprocal(1.0 + DF * mu * RLMX)[0,:]
              RayDZ = np.reciprocal(1.0 + DF * mu * RLMZ)[0,:]
                                   
              # Compute advection update
              stateA = solA + init0
              rhsAdv = tendency.computeAdvectionLogPLogT_Explicit(PHYS, PqPxA, DqDzA, REFS, REFG, solA, stateA[:,0], stateA[:,1], ebcDex)
                     
              # Compute internal force update
              rhsIfc, RdT = tendency.computeInternalForceLogPLogT_Explicit(PHYS, PqPxA, DqDzA, REFS, REFG, solA)

              # Store the dynamic RHS
              rhsDyn = (rhsAdv + rhsIfc)
              rhsDyn = tendency.enforceTendencyBC(rhsDyn, zeroDex, ebcDex, REFS[6][0])
              
              #%% Compute diffusive update
              
              Psr = init0[:,2] * (1.0 + np.expm1(solA[:,2]))
              Rho = np.expand_dims(Psr / RdT, axis=1)
              invRho = np.reciprocal(Rho)

              # Compute directional derivative along terrain
              PqPxA[bdex,:] = S * DqDxA[bdex,:]
              
              if diffusiveFlux:
                     PqPxA *= DCF[0]
                     DqDzA *= DCF[1]
                     PqPxA *= Rho
                     DqDzA *= Rho
                            
              # Compute derivatives of diffusive flux
              P2qPx2, P2qPz2, P2qPzx, P2qPxz = \
              tendency.computeFieldDerivatives2(PqPxA, DqDzA, DDXM_B, DDZM_B, REFS)
              
              # Second directional derivatives (of the diffusive fluxes)
              P2qPx2[bdex,:] += dhdx * P2qPxz[bdex,:]; P2qPx2[bdex,:] *= S
              P2qPzx[bdex,:] += dhdx * P2qPz2[bdex,:]; P2qPzx[bdex,:] *= S
              
              # Compute diffusive tendencies
              rhsDif = tendency.computeDiffusionTendency(P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                               REFS, REFG, ebcDex, DLD, DCF, diffusiveFlux)
              rhsDif = tendency.enforceTendencyBC(rhsDif, zeroDex, ebcDex, REFS[6][0])
              
              # Apply update
              rhsDif *= invRho
              solB = sol2Update + DF * (rhsDyn + rhsDif)
              #'''
              # Apply Rayleigh damping layer implicitly
              #state = solB + init0
              solB[:,0] *= RayDX.T
              solB[:,1] *= RayDZ.T
              
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
       
       def ssprk43(sol):
              rhs = 0.0
              # Stage 1
              sol1, rhs = computeUpdate(0.5, sol, sol)
              rhs += rhs
              # Stage 2
              sol2, rhs = computeUpdate(0.5, sol1, sol1)
              rhs += rhs
              
              # Stage 3 from SSPRK32
              sols = 1.0 / 3.0 * sol + 2.0 / 3.0 * sol2
              sol3, rhs = computeUpdate(1.0 / 3.0, sol2, sols)
              rhs += rhs
              
              sols = 0.5 * (sol3 + sol)
              # Stage 3
              #sols, rhs, res = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              #sol3, rhs, res = computeUpdate(1.0 / 6.0, sol2, sols, rhs)
              
              # Stage 4
              sol4, rhs = computeUpdate(0.5, sols, sols)
              rhs += rhs
              
              res = sol4 - sol3
              rhsAvg = 0.25 * rhs
                                          
              return sol4, rhsAvg
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol, rhs = computeUpdate(c1, sol, sol)
              sol1 = np.copy(sol)
              
              for ii in range(5):
                     sol, rhs = computeUpdate(c1, sol, sol)
                     
              # Compute stage 6* with linear combination
              sol = 0.6 * sol1 + 0.4 * sol
              
              for ii in range(3):
                     sol, rhs = computeUpdate(c1, sol, sol)
                            
              return sol
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol1 = np.copy(sol)
              sol2 = np.copy(sol)
              
              sol1, rhs = computeUpdate(c1, sol1, sol1)
              
              for ii in range(3):
                     sol1, rhs = computeUpdate(c1, sol1, sol1)
              
              sol2 = 0.04 * sol2 + 0.36 * sol1
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1, rhs = computeUpdate(c1, sol1, sol1)
                     
              sol = sol2 + 0.6 * sol1
              sol, rhs = computeUpdate(0.1, sol1, sol)
              
              return sol
       
       def ssprk54(sol):
              
              # Stage 1 predictor
              b10 = 0.391752226571890
              sol1, rhs = computeUpdate(b10, sol, sol)
              # Stage 1 corrector
              solp = sol + 0.5 * (sol1 - sol)
              sol1, rhs = computeUpdate(0.5 * b10, sol1, solp)
              
              # Stage 2
              a0 = 0.444370493651235
              a1 = 0.555629506348765
              sols = a0 * sol + a1 * sol1
              b21 = 0.368410593050371
              sol2, rhs = computeUpdate(b21, sol1, sols)
              
              # Stage 3
              a0 = 0.620101851488403
              a2 = 0.379898148511597
              sols = a0 * sol + a2 * sol2
              b32 = 0.251891774271694
              sol3, rhs = computeUpdate(b32, sol2, sols)
              
              # Stage 4
              a0 = 0.178079954393132
              a3 = 0.821920045606868
              sols = a0 * sol + a3 * sol3
              b43 = 0.544974750228521
              sol4, rhs = computeUpdate(b43, sol3, sols)
              fun3 = (sol4 - sols) / b43
              
              # Stage 5
              a2 = 0.517231671970585
              a3 = 0.096059710526147
              a4 = 0.386708617503269
              b53 = 0.063692468666290
              b54 = 0.226007483236906
              sols = a2 * sol2 + a3 * sol3 + a4 * sol4
              funs = b53 * fun3
              sol5, rhs = computeUpdate(b54, sol4, sols + funs)
              
              return sol5
       
       def ssprk63(sol):
              
              # Stage 1
              b10 = 0.284220721334261
              sol1, rhs = computeUpdate(b10, sol, sol)
              
              # Stage 2
              b21 = 0.284220721334261
              sol2, rhs = computeUpdate(b21, sol1, sol1)
              
              # Stage 3
              b32 = 0.284220721334261
              sol3, rhs = computeUpdate(b32, sol2, sol2)
              
              # Stage 4
              a0 = 0.476769811285196
              a1 = 0.098511733286064
              a3 = 0.424718455428740
              sols = a0 * sol + a1 * sol1 + a3 * sol3
              b43 =  0.120713785765930
              sol4, rhs = computeUpdate(b43, sol3, sols)
              
              # Stage 5
              b54 = 0.284220721334261
              sol5, rhs = computeUpdate(b54, sol4, sol4)
              
              # Stage 6
              a2 = 0.155221702560091
              a5 = 0.844778297439909
              sols = a2 * sol2 + a5 * sol5
              b65 =  0.240103497065900
              sol6, rhs = computeUpdate(b65, sol5, sols)
              
              return sol6

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              solB = ketchesonM2(sol0)
       elif order == 3:
              solB = ssprk63(sol0) #ketcheson93(sol0)
       elif order == 4:
              #solB = ssprk54(sol0) 
              solB = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              solB = ketchesonM2(sol0)
       
       return solB