#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
import numpy as np
import computeEulerEquationsLogPLogT as tendency

def computeTimeIntegrationLN(REFS, bN, AN, DT, RHS, SOLT, RESCF, sysDex, udex, wdex, pdex, tdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(1,6):
              # Update the solution
              SOLT[sysDex,0] += c1 * DT * RHS
              if ii == 1:
                     # Copy to storage 2
                     SOLT[sysDex,1] = SOLT[sysDex,0]
              # Update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0])
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              
       # Compute stage 6 with linear combination
       SOLT[sysDex,0] = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * \
              (SOLT[sysDex,0] + c1 * DT * RHS))
       
       # Update the RHS
       #RHS = bN - AN.dot(SOLT[sysDex,0])
       #RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              # Update the solution
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0])
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              
       return SOLT, RHS

def computeTimeIntegrationNL(PHYS, REFS, REFG, DT, SOLT, RHS, INIT, RESCF, sysDex, udex, wdex, pdex, tdex, ubdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       #'''
       #%% THE KETCHENSON SSP(9,3) METHOD
       # Compute stages 1 - 5
       for ii in range(1,6):
              SOLT[sysDex,0] += c1 * DT * RHS
              if ii == 1:
                     # Copy to storage 2
                     SOLT[sysDex,1] = SOLT[sysDex,0]       
              
              # Update the RHS
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              #RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              
       # Compute stage 6 with linear combination
       SOLT[sysDex,0] = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * \
              (SOLT[sysDex,0] + c1 * DT * RHS))
       
       # Update the RHS
       #RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
       #RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       #RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              #RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       #'''
       
       #%% Optimal SSPRK(5,3)
       '''
       alpha = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0],
               [0.56656131914033, 0.0, 0.43343868085967, 0.0, 0.0],
               [0.09299483444413, 0.00002090369620, 0.0, 0.90698426185967, 0.0],
               [0.00736132260920, 0.20127980325145, 0.00182955389682, 0.78952932024253]])
       
       beta = np.array([0.37726891511710, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.37726891511710, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.16352294089771, 0.0, 0.0],
                       [0.00071997378654, 0.0, 0.0, 0.34217696850008, 0.0],
                       [0.00277719819460, 0.00001567934613, 0.0, 0.0, 0.29786487010104])
       
       for ii in range(5):
              for kk in range(ii):
                     SOLT[sysDex,0] = alpha[ii,kk] * SOLT[sysDex,0] + beta[ii,kk] * DT * RHS
       
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       '''
       return SOLT, RHS