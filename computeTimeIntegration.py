#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import matplotlib.pyplot as plt
import computeResidualViscCoeffs as rescf
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
       
def computeTimeIntegrationNL(PHYS, REFS, REFG, DLD, TOPT, \
                             sol0, init0, rhs0, dsol0, CRES, ebcDex, \
                             RSBops, VWAV_ref, sol_norm, res_norm, isInitialStep):
       
       OPS = sol0.shape[0]
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       DQDZ = REFG[2]
       
       RLM = REFG[4][0]
       RLMX = REFG[4][1]
       RLMG = REFG[4][2]
       GMLX = REFG[0][1]
       GMLZ = REFG[0][2]
       
       S = DLD[5]
       dhdx = np.expand_dims(REFS[6][0], axis=1)
       bdex = ebcDex[2]

       # Stacked derivative operators
       DD1 = REFS[12] # First derivatives for advection/dynamics
       DD2 = REFS[13] # First derivatives for diffusion gradient
       
       rhs = 0.0
       res = 0.0
       DynSGS_Update = True
       Residual_Update = True
       Timestep_Update = True
       
       def computeUpdate(coeff, solA, sol2Update):
              
              nonlocal DynSGS_Update,Residual_Update,Timestep_Update,\
                       rhs,res,CRES,DT
              
              # Compute time dependent state about initial conditions
              pertbA = solA - init0
              
              # Compute pressure gradient force scaling (buoyancy)
              RdT, T_ratio = tendency.computeRdT(pertbA[:,2], pertbA[:,3],
                                                 REFS[9][0], PHYS[4])
              
              #%% First dynamics update
              if RSBops:
                     Dq = DD1.dot(solA)
              else:
                     Dq = DD1 @ solA
                     
              DqDxA = Dq[:OPS,:]
              PqPzA = Dq[OPS:,:]
              
              PqPxA = DqDxA - REFS[14] * PqPzA
              DqDzA = (PqPzA - DQDZ)
                                   
              # Compute local RHS
              rhsDyn = tendency.computeEulerEquationsLogPLogT_Explicit(PHYS, PqPxA, PqPzA, DqDzA, 
                                                                       RdT, T_ratio, solA)
              rhsDyn = tendency.enforceBC_RHS(rhsDyn, ebcDex)
              
              if Residual_Update:
                     rhs = np.copy(rhsDyn)
                     #res = dsol0 / TOPT[0] - 0.5 * (rhs0 + rhs)
                     res = 2.0 * (rhs - 0.5 * (rhs0 + rhs))
                     res *= res_norm
                     Residual_Update = False
                     
              if Timestep_Update:
                     
                     DT, VWAV_fld, VFLW_adv = tendency.computeNewTimeStep(PHYS, RdT, solA,
                                                                          DLD, isInitial=isInitialStep)

                     Timestep_Update = False
              
              #%% Compute the DynSGS coefficients at the top update
              if DynSGS_Update:
                     # Define residual as the timestep change in the RHS
                     CRES = rescf.computeResidualViscCoeffs(np.abs(res), sol_norm, DLD,
                                                            bdex, REFG[5], RLM, VWAV_ref, CRES)
       
                     DynSGS_Update = False
              
              #%% Compute diffusive update

              # Compute directional derivative along terrain
              PqPxA[bdex,:] = S * DqDxA[bdex,:]
              PqPzA = np.copy(DqDzA)
              
              # Compute diffusive fluxes
              PqPxA *= CRES[:,0,:]
              PqPzA *= CRES[:,1,:]
              
              # Compute derivatives of diffusive flux
              Dq = np.column_stack((PqPxA,PqPzA))
              if RSBops:
                     DDq = DD2.dot(Dq)
              else:
                     DDq = DD2 @ Dq
                     
              # Column 1
              D2qDx2 = DDq[:OPS,:4]
              P2qPxz = DDq[OPS:,:4]
              # Column 2
              D2qDzx = DDq[:OPS,4:]
              P2qPz2 = DDq[OPS:,4:]
              
              P2qPx2 = D2qDx2 - REFS[14] * P2qPxz
              P2qPzx = D2qDzx - REFS[14] * P2qPz2
              
              # Second directional derivatives (of the diffusive fluxes)
              P2qPx2[bdex,:] = S * D2qDx2[bdex,:]
              P2qPzx[bdex,:] = S * D2qDzx[bdex,:]
              
              # Compute diffusive tendencies
              rhsDif = tendency.computeDiffusionTendency(P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                                         ebcDex)
              rhsDif = tendency.enforceBC_RHS(rhsDif, ebcDex)
              
              # Compute total RHS and apply BC
              rhs = (rhsDyn + rhsDif)
              
              # Apply stage update
              solB = sol2Update + coeff * DT * rhs
              
              # Rayleigh factor to inflow boundary
              RayD = np.reciprocal(1.0 + coeff * DT * mu * RLM)
              #RayDX = np.reciprocal(1.0 + coeff * DT * mu * RLMX)
              # Apply Rayleigh damping layer implicitly
              solB[:,0] = RayD * solB[:,0] + (1.0 - RayD) * init0[:,0]
              solB[:,1] *= RayD
              solB[:,2] = RayD * solB[:,2] + (1.0 - RayD) * init0[:,2]
              solB[:,3] = RayD * solB[:,3] + (1.0 - RayD) * init0[:,3]
              
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
       
       def ssprk43(sol):
              # Stage 1
              sol1 = computeUpdate(0.5, sol, sol)
              # Stage 2
              sol2 = computeUpdate(0.5, sol1, sol1)
              
              # Stage 3 from SSPRK32
              sols = 1.0 / 3.0 * sol + 2.0 / 3.0 * sol2
              sol3 = computeUpdate(1.0 / 3.0, sol2, sols)
              
              sols = 0.5 * (sol3 + sol)
              # Stage 3
              #sols, rhs, res = np.array(2.0 / 3.0 * sol + 1.0 / 3.0 * sol2)
              #sol3, rhs, res = computeUpdate(1.0 / 6.0, sol2, sols, rhs)
              
              # Stage 4
              sol4 = computeUpdate(0.5, sols, sols)
                                          
              return sol4
       
       def ketcheson93(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol = computeUpdate(c1, sol, sol)
              sol1 = np.copy(sol)
              
              for ii in range(5):
                     sol = computeUpdate(c1, sol, sol)
                     
              # Compute stage 6* with linear combination
              sol = 0.6 * sol1 + 0.4 * sol
              
              for ii in range(3):
                     sol = computeUpdate(c1, sol, sol)
                            
              return sol
       
       def ketcheson104(sol):
              # Ketchenson, 2008 10.1137/07070485X
              c1 = 1.0 / 6.0
              
              sol1 = np.copy(sol)
              sol2 = np.copy(sol)
              
              sol1 = computeUpdate(c1, sol1, sol1)
              
              for ii in range(3):
                     sol1 = computeUpdate(c1, sol1, sol1)
              
              sol2 = 0.04 * sol2 + 0.36 * sol1
              sol1 = 15.0 * sol2 - 5.0 * sol1
              
              for ii in range(4):
                     sol1 = computeUpdate(c1, sol1, sol1)
                     
              sol = sol2 + 0.6 * sol1
              sol = computeUpdate(0.1, sol1, sol)
              
              return sol
       
       def ssprk54(sol):
              
              # Stage 1 predictor
              b10 = 0.391752226571890
              sol1 = computeUpdate(b10, sol, sol)
              # Stage 1 corrector
              solp = sol + 0.5 * (sol1 - sol)
              sol1  = computeUpdate(0.5 * b10, sol1, solp)
              
              # Stage 2
              a0 = 0.444370493651235
              a1 = 0.555629506348765
              sols = a0 * sol + a1 * sol1
              b21 = 0.368410593050371
              sol2 = computeUpdate(b21, sol1, sols)
              
              # Stage 3
              a0 = 0.620101851488403
              a2 = 0.379898148511597
              sols = a0 * sol + a2 * sol2
              b32 = 0.251891774271694
              sol3 = computeUpdate(b32, sol2, sols)
              
              # Stage 4
              a0 = 0.178079954393132
              a3 = 0.821920045606868
              sols = a0 * sol + a3 * sol3
              b43 = 0.544974750228521
              sol4 = computeUpdate(b43, sol3, sols)
              fun3 = (sol4 - sols) / b43
              
              # Stage corrector
              solp = sols + 0.5 * (sol4 - sols)
              sol4 = computeUpdate(0.5 * b43, sol4, solp)
              
              # Stage 5
              a2 = 0.517231671970585
              a3 = 0.096059710526147
              a4 = 0.386708617503269
              b53 = 0.063692468666290
              b54 = 0.226007483236906
              sols = a2 * sol2 + a3 * sol3 + a4 * sol4 + b53 * fun3
              sol5 = computeUpdate(b54, sol4, sols)
              
              return sol5
       
       def ssprk63(sol):
              
              # Stage 1
              b10 = 0.284220721334261
              sol1 = computeUpdate(b10, sol, sol)
              # Stage 1 corrector
              solp = sol + 0.5 * (sol1 - sol)
              sol1 = computeUpdate(0.5 * b10, sol1, solp)
              
              # Stage 2
              b21 = 0.284220721334261
              sol2 = computeUpdate(b21, sol1, sol1)
              
              # Stage 3
              b32 = 0.284220721334261
              sol3 = computeUpdate(b32, sol2, sol2)
              
              # Stage 4
              a0 = 0.476769811285196
              a1 = 0.098511733286064
              a3 = 0.424718455428740
              sols = a0 * sol + a1 * sol1 + a3 * sol3
              b43 =  0.120713785765930
              sol4 = computeUpdate(b43, sol3, sols)
              
              # Stage 5
              b54 = 0.284220721334261
              sol5 = computeUpdate(b54, sol4, sol4)
              
              # Stage 6
              a2 = 0.155221702560091
              a5 = 0.844778297439909
              sols = a2 * sol2 + a5 * sol5
              b65 =  0.240103497065900
              sol6 = computeUpdate(b65, sol5, sols)
       
              return sol6

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              sol = ketchesonM2(sol0)
       elif order == 3:
              #sol1 = ssprk63(sol0) 
              sol = ketcheson93(sol0)
       elif order == 4:
              #sol1 = ssprk54(sol0) 
              sol = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              sol = ketchesonM2(sol0)
              
       return sol, sol-sol0, rhs, res, CRES, DT
