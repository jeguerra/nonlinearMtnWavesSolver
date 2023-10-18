#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import torch
import math as mt
import matplotlib.pyplot as plt
import bottleneck as bn
import computeResidualViscCoeffs as rescf
import computeEulerEquationsLogPLogT as tendency

# Change floating point errors
np.seterr(all='ignore', divide='raise', over='raise', invalid='raise')

torch.set_num_threads(12)
torch.set_num_interop_threads(8)

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
                              sol0, init0, rhs0, CRES, ebcDex, RSBops):
       
       OPS = DIMS[-2]
       DT = TOPT[0]
       order = TOPT[3]
       mu = REFG[3]
       DQDZ = REFG[2]
       RLMX = REFG[4][1].data
       RLMZ = REFG[4][2].data
       RLM = REFG[4][0].data
       
       S = DLD[5]
       dhdx = np.expand_dims(REFS[6][0], axis=1)
       
       bdex = ebcDex[2]

       # Stacked derivative operators
       DD1 = REFS[12] # First derivatives for advection/dynamics
       DD2 = REFS[13] # First derivatives for diffusion gradient
       
       rhs1 = 0.0
       res = 0.0
       DynSGS_Update = True
       Residual_Update = True
       
       def computeUpdate(coeff, solA, sol2Update):
              
              nonlocal DynSGS_Update, Residual_Update,rhs1,res,CRES,DT
              
              # Compute total state ()
              stateA = np.copy(solA)
              stateA[:,2:] += init0[:,2:]
              stateA[:,2:] = np.exp(stateA[:,2:])
              
              # Compute pressure gradient force scaling (buoyancy)
              RdT, T_ratio = tendency.computeRdT(solA, REFS[9][0], PHYS[4])
              
              #%% First dynamics update
              if RSBops:
                     Dq = DD1.dot(solA)
              else:
                     Dq = torch.matmul(DD1, torch.from_numpy(solA)).numpy()
                     
              DqDxA = Dq[:DIMS[5],:]
              DqDzA = Dq[DIMS[5]:,:]
              
              # Complete advective partial derivatives
              PqPxA = DqDxA - REFS[15] * DqDzA
                                   
              # Compute local RHS
              rhsDyn = tendency.computeEulerEquationsLogPLogT_Explicit(PHYS, PqPxA, DqDzA, DQDZ, RdT, T_ratio, solA, stateA)

              # Store the dynamic RHS
              rhsDyn = tendency.enforceBC_RHS(rhsDyn, ebcDex)
              
              if Residual_Update:
                     res = rhs0 - 0.5 * (rhs0 + rhsDyn)
                     res[:,2:] *= stateA[:,2:]
                     Residual_Update = False
              
              #%% Compute the DynSGS coefficients at the top update
              if DynSGS_Update:
                     
                     DT, VWAV_fld, VWAV_max = tendency.computeNewTimeStep(PHYS, RdT, sol0, DLD, DT)
                     
                     # compute function average of abs(variables)
                     sol_avg = np.expand_dims(DLD[-3] @ stateA, axis=0)
                            
                     # State from local average state
                     dsol = np.abs(stateA - sol_avg)
                     #sol_norm = np.nanquantile(dsol, 0.95, axis=0)
                     sol_norm = DLD[-3] @ dsol
                     
                     # Define residual as the timestep change in the RHS
                     CRES, res_norm = rescf.computeResidualViscCoeffs(DIMS, res, dsol, sol_norm, DLD, \
                                                            bdex, REFG[5], RLM, VWAV_max, CRES)
                     rhs1 = np.copy(rhsDyn)
                     res *= (1.0 / res_norm)
       
                     DynSGS_Update = False
              
              #%% Compute diffusive update
              
              PqPxA[:,2:] *= stateA[:,2:]
              DqDzA[:,2:] *= stateA[:,2:]

              # Compute directional derivative along terrain
              PqPxA[bdex,:] = S * DqDxA[bdex,:]
              
              # Compute diffusive fluxes
              PqPxA *= CRES[:,0,:]
              DqDzA *= CRES[:,1,:]
              
              # Compute derivatives of diffusive flux
              Dq = np.column_stack((PqPxA,DqDzA))
              if RSBops:
                     DDq = DD2.dot(Dq)
              else:
                     DDq = torch.matmul(DD2, torch.from_numpy(Dq)).numpy()
                     
              # Column 1
              P2qPx2 = DDq[:OPS,:4]
              P2qPzx = DDq[OPS:,:4]
              # Column 2
              P2qPxz = DDq[:OPS,4:]
              P2qPz2 = DDq[OPS:,4:]
              
              # Second directional derivatives (of the diffusive fluxes)
              P2qPx2[bdex,:] += dhdx * P2qPzx[bdex,:]; P2qPx2[bdex,:] *= S
              P2qPzx[bdex,:] += dhdx * P2qPz2[bdex,:]; P2qPzx[bdex,:] *= S
              
              # Compute diffusive tendencies
              rhsDif = tendency.computeDiffusionTendency(P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                                         ebcDex, DLD)
              rhsDif = tendency.enforceBC_RHS(rhsDif, ebcDex)
              # Compute total RHS and apply BC
              rhsDif[:,2:] *= 1.0 / stateA[:,2:]
              rhs = (rhsDyn + rhsDif)
              
              # Apply stage update
              solB = sol2Update + coeff * DT * rhs
              
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
              sol1 = ketchesonM2(sol0)
       elif order == 3:
              #sol1 = ssprk63(sol0) 
              sol1 = ketcheson93(sol0)
       elif order == 4:
              #sol1 = ssprk54(sol0) 
              sol1 = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              sol1 = ketchesonM2(sol0)
              
       # Compute local Rayleigh factors
       #RayDX = np.reciprocal(1.0 + DT * mu * RLMX)[0,:]
       #RayDZ = np.reciprocal(1.0 + DF * mu * RLMZ)[0,:]
       RayD = np.reciprocal(1.0 + DT * mu * RLM)[0,:]              

       # Apply Rayleigh damping layer implicitly
       sol1[:,0] = RayD.T * sol1[:,0] + (1.0 - RayD.T) * init0[:,0]
       sol1[:,1] *= RayD.T
       sol1[:,2] *= RayD.T
       sol1[:,3] *= RayD.T
              
       return sol1, rhs1, res, CRES, DT