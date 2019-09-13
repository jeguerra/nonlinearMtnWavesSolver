#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import computeEulerEquationsLogPLogT as tendency

def computePrepareFields(PHYS, REFS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex):
       # Get some physical quantities
       P0 = PHYS[1]
       Rd = PHYS[3]
       kap = PHYS[4]
       
       # Get the boundary terrain
       DZT = REFS[6]
       
       # Get the solution components
       uxz = SOLT[udex]
       wxz = SOLT[wdex]
       pxz = SOLT[pdex]
       txz = SOLT[tdex]
       
       # Make the total quatities
       U = uxz + INIT[udex]
       LP = pxz + INIT[pdex]
       LT = txz + INIT[tdex]
       
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       # Apply boundary condition
       wxz[botdex] = DZT[0,:] * U[botdex]
       wxz[topdex] = np.zeros(len(topdex))
       
       # Potential temperature perturbation vanishes along top boundary       
       txz[topdex] = np.zeros(len(topdex))
       
       return uxz, wxz, pxz, txz, U, RdT

def computeTimeIntegrationLN(PHYS, REFS, bN, AN, DT, RHS, SOLT, INIT, RESCF, sysDex, udex, wdex, pdex, tdex, botdex, topdex, DynSGS): 
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       sol = SOLT[sysDex,0]
       
       def computeRHSUpdate():
              rhs = bN - AN.dot(sol)
              if DynSGS:
                     uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
                     rhs += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
                     
              return rhs
       
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(7):
              sol += c1 * DT * RHS
              if ii == 1:
                     SOLT[sysDex,1] = sol
                     RHS = computeRHSUpdate()
              
       # Compute stage 6 with linear combination
       sol = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * sol)
       
       # Compute stages 7 - 9
       for ii in range(2):
              sol += c1 * DT * RHS
              RHS = computeRHSUpdate()
              
       return sol, RHS

def computeTimeIntegrationNL(PHYS, REFS, REFG, DT, bN, RHS, SOLT, INIT, RESCF, udex, wdex, pdex, tdex, botdex, topdex, DynSGS):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       sol = SOLT[:,0]
       
       def computeRHSUpdate():
              uxz, wxz, pxz, txz, U, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, RdT, botdex, topdex)
              rhs += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              if DynSGS:
                     rhs += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
                     
              return rhs
       #'''
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(7):
              sol += c1 * DT * RHS
              if ii == 1:
                     SOLT[:,1] = sol
              
              RHS = computeRHSUpdate()
              
       # Compute stage 6 with linear combination
       sol = c2 * (3.0 * SOLT[:,1] + 2.0 * sol)
       
       # Compute stages 7 - 9
       for ii in range(2):
              sol += c1 * DT * RHS
              RHS = computeRHSUpdate()
       #'''
       
       #%% THE KETCHENSON SSP(10,4) METHOD
       '''
       SOLT[:,1] = SOLT[:,0]
       for ii in range(1,6):
              sol += c1 * DT * RHS
              RHS = computeRHSUpdate()
       
       SOLT[:,1] = 0.04 * SOLT[:,1] + 0.36 * sol
       sol = 15.0 * SOLT[:,1] - 5.0 * sol
       
       for ii in range(6,10):
              sol += c1 * DT * RHS
              RHS = computeRHSUpdate()
              
       sol = SOLT[:,1] + 0.6 * sol + 0.1 * DT * RHS
       computeRHSUpdate()
       '''
       
       return sol, RHS