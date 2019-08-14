#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:09:52 2019

@author: jorge.guerra
"""

def computeTimeIntegration(bN, AN, DT, RHS, SOLT, sysDex):
       # Set the coefficients
       c1 = 1.0 / 6.0
       c2 = 1.0 / 5.0
       # Stage 1
       SOLT[sysDex,0] += c1 * DT * (RHS + SOLT[sysDex,2])
       # Copy to storage 2
       SOLT[sysDex,1] = SOLT[sysDex,0]
       
       # Compute stages 2 - 5
       for ii in range(2,6):
              SOLT[sysDex,0] += c1 * DT * RHS
              # Update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0]) + SOLT[sysDex,2]
              
       # Compute stage 6 with linear combination
       SOLT[sysDex,0] = 3.0 * SOLT[sysDex,1] + 2.0 * \
              (SOLT[sysDex,0] + c1 * DT * RHS)
       SOLT[sysDex,0] *= c2
       
       # Compute stages 7 - 9
       for ii in range(7,10):
              SOLT[sysDex,0] += c1 * DT * RHS
              # update the RHS
              RHS = bN - AN.dot(SOLT[sysDex,0]) + SOLT[sysDex,2]
              
       return SOLT, RHS