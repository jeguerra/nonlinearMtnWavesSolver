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
       U = np.add(uxz, INIT[udex])
       LP = np.add(pxz, INIT[pdex])
       LT = np.add(txz, INIT[tdex])
       
       # Compute the sensible temperature scaling to PGF
       RdT = Rd * P0**(-kap) * np.exp(LT + kap * LP)
       
       # Apply boundary condition
       wxz[botdex] = DZT[0,:] * U[botdex]
       wxz[topdex] = np.zeros(len(topdex))
       
       # Potential temperature perturbation vanishes along top boundary       
       txz[topdex] = np.zeros(len(topdex))
       
       return uxz, wxz, pxz, txz, U, LP, LT, RdT

def computeRHSUpdate():

def computeTimeIntegrationLN(PHYS, REFS, bN, AN, DT, RHS, SOLT, INIT, RESCF, sysDex, udex, wdex, pdex, tdex, botdex, topdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       
       sol = SOLT[sysDex,0]
       
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(7):
              # Update the solution
              sol += c1 * DT * RHS
              if ii == 1:
                     # Copy to storage 2
                     SOLT[sysDex,1] = sol
              # Update the RHS
              RHS = bN - AN.dot(sol)
              uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              
       # Compute stage 6 with linear combination
       sol = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * sol)
       
       # Compute stages 7 - 9
       for ii in range(2):
              # Update the solution
              sol += c1 * DT * RHS
              # update the RHS
              RHS = bN - AN.dot(sol)
              uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              
       return sol, RHS

def computeTimeIntegrationNL(PHYS, REFS, REFG, DT, bN, RHS, SOLT, INIT, RESCF, udex, wdex, pdex, tdex, botdex, topdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       sol = SOLT[:,0]
       #'''
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(7):
              # Update the solution
              sol += c1 * DT * RHS
              if ii == 1:
                     # Copy to storage 2
                     SOLT[:,1] = sol
              # Update the RHS
              uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, LP, LT, RdT, botdex, topdex)
              RHS += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              
       # Compute stage 6 with linear combination
       sol = c2 * (3.0 * SOLT[:,1] + 2.0 * sol)
       
       # Compute stages 7 - 9
       for ii in range(2):
              sol += c1 * DT * RHS
                     
              # Update the RHS
              uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, LP, LT, RdT, botdex, topdex)
              RHS += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       #'''
       
       #%% THE KETCHENSON SSP(10,4) METHOD
       '''
       SOLT[:,1] = SOLT[:,0]
       for ii in range(1,6):
              # Update the solution
              sol += c1 * DT * RHS
              # Update the RHS
              uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, LP, LT, RdT, botdex, topdex)
              RHS += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              #RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       
       SOLT[:,1] = 0.04 * SOLT[:,1] + 0.36 * sol
       sol = 15.0 * SOLT[:,1] - 5.0 * sol
       
       for ii in range(6,10):
              # Update the solution
              sol += c1 * DT * RHS
              # Update the RHS
              uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, LP, LT, RdT, botdex, topdex)
              RHS += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              #RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
              
       sol = SOLT[:,1] + 0.6 * sol + 0.1 * DT * RHS
       # Update the RHS
       uxz, wxz, pxz, txz, U, LP, LT, RdT = computePrepareFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
       RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, uxz, wxz, pxz, txz, U, LP, LT, RdT, botdex, topdex)
       RHS += tendency.computeRayleighTendency(REFG, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       #RHS += tendency.computeDynSGSTendency(RESCF, REFS, uxz, wxz, pxz, txz, udex, wdex, pdex, tdex, botdex, topdex)
       '''
       return sol, RHS