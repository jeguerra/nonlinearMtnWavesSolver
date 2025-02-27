#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import torch as tch
import bottleneck as bn
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
       
def computeTimeIntegrationNL(PHYS, REFS, REFG, DLD, DT, TOPT, \
                             sol0, init0, rhs0, dsol0, dcf0, ebcDex, \
                             RSBops, VWAV_ref, res_norm, \
                             isInitialStep, DynSGS_RES):
       
       dcf1 = 0.0
       rhsDyn = 0.0
       rhsDif = 0.0
       resVec = 0.0
       
       OPS = sol0.shape[0]
       order = TOPT[3]
       DQDZ = REFG[3]
       
       mu = REFG[4]
       RLML = REFG[0][0]
       RLMR = REFG[0][1]
       RLMT = REFG[0][2]
       RLMA = REFG[0][-1]
       #GMLX = REFG[0][1]
       #GMLZ = REFG[0][2]
       
       DZDX = REFS[14]
       S = REFS[15]
       bdex = ebcDex[2]

       # Stacked derivative operators
       DD1 = REFS[12] # First derivatives for advection/dynamics
       DD2 = REFS[13] # First derivatives for diffusion gradient
       
       # reference Rd * T
       RdT_bar = REFS[9][0]
       
       Auxilary_Updates = True
       
       def computeUpdate(coeff, solA, sol2Update):
              
              nonlocal Auxilary_Updates,rhsDyn,rhsDif,resVec,dcf1,DT
              
              # Compute time dependent state about initial conditions
              pertbA = solA - init0
              
              # Compute pressure gradient force scaling (buoyancy)
              RdT, T_ratio = tendency.computeRdT(PHYS, solA, pertbA, RdT_bar)
              
              #%% First dynamics update
              if tch.is_tensor(solA):
                     Dq = DD1 @ solA
              else:
                     Dq = DD1.dot(solA)
              DqDxA = Dq[:OPS,:]
              PqPzA = Dq[OPS:,:]
              PqPxA = DqDxA - DZDX * PqPzA
              DqDzA = (PqPzA - DQDZ)
               
              # Compute local RHS
              rhsDyn = tendency.computeEulerEquationsLogPLogT_Explicit(PHYS, PqPxA, PqPzA, DqDzA, 
                                                                       RdT, T_ratio, solA)
              if Auxilary_Updates:
                     
                     # Compute residual estimate
                     if DynSGS_RES:
                            resVec = (dsol0 / DT - 0.5 * (rhs0 + rhsDyn))
                     else:
                            resVec = 0.5 * (rhs0 + rhsDyn)
                     resVec = tendency.enforceBC_RHS(resVec, ebcDex)
                     
                     # Compute the new incoming time step and energy bound
                     DT, VWAV_max = tendency.computeNewTimeStep(PHYS, RdT, solA,
                                                                DLD, isInitial=isInitialStep)
                     
                     # Compute new DynSGS coefficients
                     if tch.is_tensor(resVec):
                            sbnd = (0.5 * DT * VWAV_max**2).cpu().item()
                            resVecAN = tch.max(resVec.abs() @ res_norm, 1).values.cpu().numpy()
                     else:
                            sbnd = 0.5 * DT * VWAV_max**2
                            resVecAN = bn.nanmax(np.abs(resVec) @ res_norm, axis=1)
                     
                     dcf1 = rescf.computeResidualViscCoeffs(resVecAN, 
                                                            DLD, DT, bdex, sbnd)
                                       
                     # Residual contribution vanishes in the sponge layers
                     dcf1[:,:,0] *= (1.0 - RLMA)
                     # Add in smooth diffusion field in the sponge layers
                     dcf1[:,:,0] += sbnd * RLMA
                     
                     if tch.is_tensor(resVec):
                            dcf1 = tch.tensor(dcf1, dtype=tch.double)
                            dcf1 = dcf1.to(resVec.device)
       
                     Auxilary_Updates = False
              
              #%% Compute diffusive update

              # Compute directional derivative along terrain
              PqPxA[bdex,:] = S * DqDxA[bdex,:]
              if tch.is_tensor(PqPzA):
                     PqPzA = DqDzA.clone()
              else:
                     PqPzA = np.copy(DqDzA)
              
              # Compute diffusive fluxes
              PqPxA *= dcf1[:,0,:]
              PqPzA *= dcf1[:,1,:]
              
              # Compute derivatives of diffusive flux
              if tch.is_tensor(PqPxA) and tch.is_tensor(PqPzA):
                     Dq = tch.column_stack((PqPxA,PqPzA))
                     DDq = DD2 @ Dq
              else:
                     Dq = np.column_stack((PqPxA,PqPzA))
                     DDq = DD2.dot(Dq) 
                     
              # Column 1
              D2qDx2 = DDq[:OPS,:4]
              P2qPxz = DDq[OPS:,:4]
              # Column 2
              D2qDzx = DDq[:OPS,4:]
              P2qPz2 = DDq[OPS:,4:]
              
              P2qPx2 = D2qDx2 - DZDX * P2qPxz
              P2qPzx = D2qDzx - DZDX * P2qPz2
              
              # Second directional derivatives (of the diffusive fluxes)
              P2qPx2[bdex,:] = S * D2qDx2[bdex,:]
              P2qPzx[bdex,:] = S * D2qDzx[bdex,:]
              
              # Compute diffusive tendencies
              rhsDif = tendency.computeDiffusionTendency(P2qPx2, P2qPz2, P2qPzx, P2qPxz, \
                                                         ebcDex)
              
              # Apply stage update
              solB = sol2Update + coeff * DT * (rhsDyn + rhsDif)
              solB = tendency.enforceBC_SOL(solB, ebcDex, init0)
              
              # Rayleigh factor to inflow boundary implicitly
              RayD = 1.0 / (1.0 + coeff * DT * mu * RLML)
              solB[:,1:] = RayD[:,1:] * solB[:,1:] +\
                           (1.0 - RayD)[:,1:] * init0[:,1:]

              # Rayleigh factor to outflow boundary implicitly
              RayD = 1.0 / (1.0 + coeff * DT * mu * RLMR)
              solB[:,0:2] = RayD[:,0:2] * solB[:,0:2]
              solB[:,2:] = RayD[:,2:] * solB[:,2:] +\
                           (1.0 - RayD)[:,2:] * init0[:,2:]
              
              # Rayleigh factor to top boundary implicitly
              RayD = 1.0 / (1.0 + coeff * DT * mu * RLMT)
              solB[:,1] = RayD[:,1] * solB[:,1]
              
              # Apply BC
              solB = tendency.enforceBC_SOL(solB, ebcDex, init0)
              
              return solB
       
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
              
              if tch.is_tensor(sol):
                     sol1 = sol.clone()
              else:
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
              
              if tch.is_tensor(sol):
                     sol1 = sol.clone()
                     sol2 = sol.clone()
              else:
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

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 3:
              #sol = ssprk43(sol0) 
              sol = ketcheson93(sol0)
       elif order == 4:
              #sol1 = ssprk54(sol0) 
              sol = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with SSPRK43.')
              sol = ssprk43(sol0)
           
       return sol, sol-sol0, rhsDyn, rhsDif, resVec, dcf1, DT
