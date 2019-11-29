#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import computeEulerEquationsLogPLogT as tendency
from computeResidualViscCoeffs import computeResidualViscCoeffs

def computeTimeIntegrationLN(PHYS, REFS, REFG, bN, AN, DX, DZ, DT, RHS, SOLT, INIT, sysDex, udex, wdex, pdex, tdex, botdex, topdex, DynSGS): 
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       sol = SOLT[sysDex,0]
       rhs = RHS[sysDex]
       sgs = 0.0
       def computeDynSGSUpdate():
              if DynSGS:
                     fields, wxz, U, RdT = tendency.computeUpdatedFields(PHYS, REFS, SOLT[:,0], INIT, udex, wdex, pdex, tdex, botdex, topdex)
                     RESCF = computeResidualViscCoeffs(SOLT[:,0], RHS, DX, DZ, udex, wdex, pdex, tdex)
                     rhsSGS = tendency.computeDynSGSTendency(RESCF, REFS, fields, udex, wdex, pdex, tdex, botdex, topdex)
                     rhs = rhsSGS[sysDex] 
              else:
                     rhs = 0.0
                     
              return rhs
       
       def computeRHSUpdate():
              rhs = bN - AN.dot(sol)
                 
              return rhs
       
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(7):
              sol += c1 * DT * (rhs + sgs)
              rhs = computeRHSUpdate()
              sgs = computeDynSGSUpdate()
              
              if ii == 1:
                     SOLT[sysDex,1] = sol
                     
       # Compute stage 6 with linear combination
       sol = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * sol)
       
       # Compute stages 7 - 9
       for ii in range(2):
              sol += c1 * DT * (rhs + sgs)
              rhs = computeRHSUpdate()
              sgs = computeDynSGSUpdate()
              
       return sol, rhs

def computeTimeIntegrationNL(PHYS, REFS, REFG, DX, DZ, DT, RHS, SOLT, INIT, udex, wdex, pdex, tdex, botdex, topdex, DynSGS):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       sol = SOLT[:,0]
       SGS = 0.0
       def computeDynSGSUpdate(fields):
              if DynSGS:
                     RESCF = computeResidualViscCoeffs(sol, RHS, DX, DZ, udex, wdex, pdex, tdex)
                     rhsSGS = tendency.computeDynSGSTendency(RESCF, REFS, fields, udex, wdex, pdex, tdex, botdex, topdex)
              else:
                     rhsSGS = 0.0
                     
              return rhsSGS
       
       def computeRHSUpdate(fields, U, RdT):
              rhs = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, fields, U, RdT, botdex, topdex)
              rhs += tendency.computeRayleighTendency(REFG, fields, botdex, topdex)
       
              return rhs
       #'''
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(7):
              sol += c1 * DT * (RHS + SGS)
              fields, U, RdT = tendency.computeUpdatedFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS = computeRHSUpdate(fields, U, RdT)
              #SGS = computeDynSGSUpdate(fields)
              
              if ii == 1:
                     SOLT[:,1] = sol
              
       # Compute stage 6 with linear combination
       sol = c2 * (3.0 * SOLT[:,1] + 2.0 * sol)
       
       # Compute stages 7 - 9
       for ii in range(2):
              sol += c1 * DT * (RHS + SGS)
              fields, U, RdT = tendency.computeUpdatedFields(PHYS, REFS, sol, INIT, udex, wdex, pdex, tdex, botdex, topdex)
              RHS = computeRHSUpdate(fields, U, RdT)
              SGS = computeDynSGSUpdate(fields)
              
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