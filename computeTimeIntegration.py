#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import torch
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
                             sol0, init0, rhs0, dsol0, dcf0_size, ebcDex, \
                             RSBops, VWAV_ref, res_norm, isInitialStep):
       
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
       
       S = DLD[5]
       bdex = ebcDex[2]

       # Stacked derivative operators
       DD1 = REFS[13] # First derivatives for advection/dynamics
       DD2 = REFS[12] # First derivatives for diffusion gradient
       
       Auxilary_Updates = True
       
       def computeUpdate(coeff, solA, sol2Update):
              
              nonlocal Auxilary_Updates,rhsDyn,rhsDif,resVec,dcf1,DT
              
              # Compute time dependent state about initial conditions
              pertbA = solA - init0
              
              # Compute pressure gradient force scaling (buoyancy)
              RdT, T_ratio = tendency.computeRdT(PHYS, solA, pertbA, REFS[9][0])
              
              #%% First dynamics update
              Dq = tendency.computeFieldDerivative(solA, DD1, RSBops)
              DqDxA = Dq[:OPS,:]
              PqPzA = Dq[OPS:,:]
              PqPxA = DqDxA - REFS[14] * PqPzA
              DqDzA = (PqPzA - DQDZ)
               
              # Compute local RHS
              rhsDyn = tendency.computeEulerEquationsLogPLogT_Explicit(PHYS, PqPxA, PqPzA, DqDzA, 
                                                                       RdT, T_ratio, solA)
              if Auxilary_Updates:
                     
                     # Compute the new incoming time step and energy bound
                     DT, VWAV_fld, VFLW_adv = tendency.computeNewTimeStep(PHYS, RdT, solA,
                                                                          DLD, isInitial=isInitialStep)
                     sbnd = 0.5 * DT * VWAV_ref**2
                     
                     # Compute residual estimate
                     #res = 0.5 * (rhs0 + rhsDyn)
                     resVec = 0.5 * ((rhsDyn - rhs0) + (dsol0 / DT - 0.5 * (rhs0 + rhsDyn)))
                     
                     # Compute new DynSGS coefficients
                     dcf1 = rescf.computeResidualViscCoeffs(res_norm, 
                                                            np.abs(resVec), 
                                                            DLD, DT, 
                                                            bdex, sbnd,
                                                            np.zeros(dcf0_size))
                                       
                     # Residual contribution vanishes in the sponge layers
                     dcf1[:,:,0] *= (1.0 - RLMA)
                     # Add in smooth diffusion field in the sponge layers
                     dcf1[:,:,0] += sbnd * RLMA
       
                     Auxilary_Updates = False
              
              #%% Compute diffusive update

              # Compute directional derivative along terrain
              PqPxA[bdex,:] = S * DqDxA[bdex,:]
              PqPzA = np.copy(DqDzA)
              
              # Compute diffusive fluxes
              PqPxA *= dcf1[:,0,:]
              PqPzA *= dcf1[:,1,:]
              
              # Compute derivatives of diffusive flux
              Dq = np.column_stack((PqPxA,PqPzA))
              DDq = tendency.computeFieldDerivative(Dq, DD2, RSBops)
                     
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
              
              # Apply stage update
              solB = sol2Update + coeff * DT * (rhsDyn + rhsDif)
              solB = tendency.enforceBC_SOL(solB, ebcDex, init0)
              
              # Rayleigh factor to inflow boundary implicitly
              RayD = np.reciprocal(1.0 + coeff * DT * mu * RLML)
              solB[:,1:] = RayD[:,1:] * solB[:,1:] +\
                           (1.0 - RayD)[:,1:] * init0[:,1:]

              # Rayleigh factor to outflow boundary implicitly
              RayD = np.reciprocal(1.0 + coeff * DT * mu * RLMR)
              solB[:,0:2] = RayD[:,0:2] * solB[:,0:2]
              solB[:,2:] = RayD[:,2:] * solB[:,2:] +\
                           (1.0 - RayD)[:,2:] * init0[:,2:]
              
              # Rayleigh factor to top boundary implicitly
              RayD = np.reciprocal(1.0 + coeff * DT * mu * RLMT)
              solB[:,1] = RayD[:,1] * solB[:,1]
              
              # Apply BC
              solB = tendency.enforceBC_SOL(solB, ebcDex, init0)
              
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

       #%% THE MAIN TIME INTEGRATION STAGES
       
       # Compute dynamics update
       if order == 2:
              sol = ketchesonM2(sol0)
       elif order == 3:
              #sol = ssprk43(sol0) 
              sol = ketcheson93(sol0)
       elif order == 4:
              #sol1 = ssprk54(sol0) 
              sol = ketcheson104(sol0)
       else:
              print('Invalid time integration order. Going with 2.')
              sol = ketchesonM2(sol0)
           
       return sol, sol-sol0, rhsDyn, rhsDif, resVec, dcf1, DT
