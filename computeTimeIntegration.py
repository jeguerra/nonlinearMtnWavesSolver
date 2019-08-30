#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""
#import numpy as np
import computeEulerEquationsLogPLogT as tendency

def computeTimeIntegrationLN(REFS, bN, AN, DT, RHS, SOLT, RESCF, sysDex, udex, wdex, pdex, tdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       
       # Compute stages 2 - 5
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
       RHS = bN - AN.dot(SOLT[sysDex,0])
       RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              # Update the solution
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0])
              if ii < 9:
                     RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              
       return SOLT, RHS

def computeTimeIntegrationNL(PHYS, REFS, REFG, DT, SOLT, RHS, INIT, RESCF, sysDex, udex, wdex, pdex, tdex, ubdex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       
       # Compute stages 1 - 5
       for ii in range(1,6):
              SOLT[sysDex,0] += c1 * DT * RHS
              if ii == 1:
                     # Copy to storage 2
                     SOLT[sysDex,1] = SOLT[sysDex,0]       
              
              # Update the RHS
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              
       # Compute stage 6 with linear combination
       SOLT[sysDex,0] = c2 * (3.0 * SOLT[sysDex,1] + 2.0 * \
              (SOLT[sysDex,0] + c1 * DT * RHS))
       
       # Update the RHS
       RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
       RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RHS = tendency.computeEulerEquationsLogPLogT_NL(PHYS, REFS, REFG, SOLT[:,0], INIT, sysDex, udex, wdex, pdex, tdex, ubdex)
              RHS += tendency.computeRayleighTendency(REFG, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              if ii < 9:
                     RHS += tendency.computeDynSGSTendency(RESCF, REFS, SOLT[:,0], sysDex, udex, wdex, pdex, tdex)
              
       return SOLT, RHS