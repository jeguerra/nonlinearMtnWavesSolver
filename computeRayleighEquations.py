#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:56:44 2019

Creates a unit normalized field of Rayleigh damping (picture frame)
Works the old fashioned way with lots of nested loops... so sue me!

@author: -
"""

import math as mt
import numpy as np
import scipy.sparse as sps
#from matplotlib import cm
#import matplotlib.pyplot as plt

def computeRayleighField(DIMS, REFS, depth, width, applyTop, applyLateral):
       
       # Get DIMS data
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       NX = DIMS[3]
       NZ = DIMS[4]
       
       # Get REFS data
       X = REFS[4]
       Z = REFS[5]
       
       # Set the layer bounds
       dLayerZ = ZH - depth
       dLayerR = L2 - width
       dLayerL = L1 + width
       
       # Assemble the Rayleigh field
       RL = np.zeros((NZ, NX))
       SBR = np.zeros((NZ, NX))
       
       for ii in range(NZ):
              for jj in range(NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     if applyLateral:
                            # Left layer or right layer or not?
                            if XRL >= dLayerR:
                                   dNormX = (L2 - XRL) / width
                            elif XRL <= dLayerL:
                                   dNormX = (XRL - L1) / width
                            else:
                                   dNormX = 1.0
                            # Evaluate the strength of the field
                            RFX = (mt.cos(0.5 * mt.pi * dNormX))**4
                     if applyTop:
                            # In the top layer?
                            if ZRL >= dLayerZ:
                                   dNormZ = (ZH - ZRL) / depth
                            else:
                                   dNormZ = 1.0
                            # Evaluate the strength of the field
                            RFZ = (mt.cos(0.5 * mt.pi * dNormZ))**4
                     
                     # Set the field to max(lateral, top) to handle corners
                     RL[ii,jj] = np.amax([RFX, RFZ])
                     # Set the binary matrix
                     if RL[ii,jj] != 0.0:
                            SBR[ii,jj] = 1.0
                            
       #plt.figure()
       #plt.contourf(X, Z, RL, 101, cmap=cm.seismic)
                            
       return RL, SBR

def computeRayleighEquations(DIMS, REFS, mu, depth, width, applyTop, applyLateral):
       # Get DIMS data
       NX = DIMS[3]
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Set up the Rayleigh field
       RL, SBR = computeRayleighField(DIMS, REFS, depth, width, applyTop, applyLateral)
       
       # Get the individual mu for each prognostic
       mu_U = mu[0]
       mu_W = mu[1]
       mu_P = mu[2]
       mu_T = mu[3]
       
       # Compute the blocks
       tempDiagonal = np.reshape(RL, (OPS,), order='F')
       RLM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       
       # Store the diagonal blocks corresponding to Rayleigh damping terms
       ROPS = [mu_U * RLM, mu_W * RLM, mu_P * RLM, mu_T * RLM]
       
       return ROPS
       
                            