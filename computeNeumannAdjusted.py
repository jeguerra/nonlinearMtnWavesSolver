#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:10:16 2019

@author: jorge.guerra
"""
import numpy as np

def computeNeumannAdjusted(DM, Left, Right):
       # Get matrix size
       N = DM.shape[0]
       DMN = np.zeros(DM.shape)
       
       #cdex = np.array(range(N))
       
       # For the left end
       if Left:
              # Loop over columns 1 to N-1
              for cc in range(1,N):
                     DMN[:,cc] = DM[:,cc] - DM[0,cc] / DM[0,0] * DM[:,0]
       elif Right:
              # Loop over columns 0 to N-2
              for cc in range(0,N-1):
                     DMN[:,cc] = DM[:,cc] - DM[N-1,cc] / DM[N-1,N-1] * DM[:,N-1]
       
       return DMN