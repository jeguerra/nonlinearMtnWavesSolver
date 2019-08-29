#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
#import numpy as np
import computeEulerEquationsLogPLogT as tendency

def computeTimeIntegrationLN(bN, AN, DT, RHS, SOLT, sysDex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       # Stage 1
       SOLT[sysDex,0] += c1 * DT * RHS
       # Copy to storage 2
       SOLT[sysDex,1] = SOLT[sysDex,0]
       
       # Compute stages 2 - 5
       for ii in range(2,6):
              # Update the solution
              SOLT[sysDex,0] += c1 * DT * RHS
              # Update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0])
              RHS += SOLT[sysDex,2]
              
       # Compute stage 6 with linear combination
       SOLT[sysDex,0] = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * \
              (SOLT[sysDex,0] + c1 * DT * RHS))
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              # Update the solution
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0])
              # Estimate the residual
              RES = RHS
              # Add in the diffusion tendency
              RHS += SOLT[sysDex,2]
              
       return SOLT, RHS, RES

def computeTimeIntegrationNL(PHYS, REFS, REFG, DT, SOLT, RHS, INIT, sysDex, udex, wdex, pdex, tdex, ubdex, wbdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       # Stage 1
       SOLT[sysDex,0] += c1 * DT * RHS
       # Copy to storage 2
       SOLT[sysDex,1] = SOLT[sysDex,0]
       
       # Compute stages 2 - 5
       for ii in range(2,6):
              SOLT[sysDex,0] += c1 * DT * RHS
              # Update the RHS
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              RHS += SOLT[sysDex,2]
              
       # Compute stage 6 with linear combination
       SOLT[sysDex,0] = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * \
              (SOLT[sysDex,0] + c1 * DT * RHS))
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RES = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS = RES + tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              # Add in the diffusion tendency
              RHS += SOLT[sysDex,2]
              
       return SOLT, RHS, RES