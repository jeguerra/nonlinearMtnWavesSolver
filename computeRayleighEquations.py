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
from matplotlib import cm
import matplotlib.pyplot as plt

def computeRayleighField(DIMS, REFS, height, width, applyTop, applyLateral, symmetricProfile):
       
       # Get DIMS data
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       
       RP = 4
       GP = 2
       
       # Get REFS data
       X = REFS[4]
       Z = REFS[5]
       
       # Set the layer bounds
       dLayerZ = height
       dLayerR = L2 - width
       dLayerL = L1 + width
       depth = ZH - height
       
       # Assemble the Rayleigh field
       RL = np.zeros((NZ, NX))
       RLX = np.zeros((NZ, NX))
       RLZ = np.zeros((NZ, NX))
       SBR = np.zeros((NZ, NX))
       
       for ii in range(0,NZ):
              for jj in range(0,NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     if applyLateral:
                            # Left layer or right layer or not? [1 0]
                            if XRL >= dLayerR:
                                   dNormX = (L2 - XRL) / width
                            elif XRL <= dLayerL:
                                   dNormX = (XRL - L1) / width
                            else:
                                   dNormX = 1.0
                            # Evaluate the Rayleigh factor
                            if symmetricProfile:
                                   RFX = mt.exp(-(2.5 * (dNormX - 0.5))**4) * \
                                          (1.0 - (mt.cos(mt.pi * dNormX))**6)
                            else:
                                   RFX = (mt.cos(0.5 * mt.pi * dNormX))**RP
                     else:
                            RFX = 0.0
                     if applyTop:
                            # In the top layer?
                            if ZRL >= dLayerZ[jj]:
                                   # This maps [depth ZH] to [1 0]
                                   dNormZ = (ZH - ZRL) / depth[jj]
                            else:
                                   dNormZ = 1.0
                            # Evaluate the strength of the field
                            if symmetricProfile:
                                   RFZ = mt.exp(-(2.5 * (dNormZ - 0.5))**4) * \
                                          (1.0 - (mt.cos(mt.pi * dNormZ))**6)
                            else:
                                   RFZ = (mt.cos(0.5 * mt.pi * dNormZ))**RP
                     else:
                            RFZ = 0.0
                     
                     # Set the field to max(lateral, top) to handle corners
                     RLX[ii,jj] = RFX
                     RLZ[ii,jj] = RFZ
                     RL[ii,jj] = np.amax([RFX, RFZ])
                     # Set the binary matrix
                     if RL[ii,jj] != 0.0:
                            SBR[ii,jj] = 1.0                            
       '''
       plt.figure()
       plt.contourf(X, Z, RL, 101, cmap=cm.seismic)
       plt.colorbar()
       plt.show()
       input()
       '''                     
       # Assemble the Grid Matching Layer field X and Z directions
       GML = np.ones((NZ, NX))
       GMLX = np.ones((NZ, NX))
       GMLZ = np.ones((NZ, NX))
       for ii in range(0,NZ):
              for jj in range(0,NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     if applyLateral:
                            # Left layer or right layer or not? [0 1]
                            if XRL >= dLayerR:
                                   dNormX = (XRL - dLayerR) / width
                            elif XRL <= dLayerL:
                                   dNormX = (dLayerL - XRL) / width
                            else:
                                   dNormX = 0.0
                            # Evaluate the GML factor
                            RFX = (mt.tan(0.5 * mt.pi * dNormX))**GP
                     else:
                            RFX = 0.0
                     if applyTop:
                            # In the top layer?
                            if ZRL >= dLayerZ[jj]:
                                   dNormZ = (ZRL - dLayerZ[jj]) / (ZH - height[jj])
                            else:
                                   dNormZ = 0.0
                            # Evaluate the strength of the field
                            RFZ = (mt.tan(0.5 * mt.pi * dNormZ))**GP
                     else:
                            RFZ = 0.0
                     
                     GMLX[ii,jj] = 1.0 / (1.0 + RFX)
                     GMLZ[ii,jj] = 1.0 / (1.0 + RFZ)
                     # Set the field to max(lateral, top) to handle corners
                     RFM = np.amax([RFX, RFZ])
                     GML[ii,jj] = 1.0 / (1.0 + RFM)
                     
       '''
       plt.figure()
       plt.contourf(X, Z, GML, 101, cmap=cm.seismic)
       plt.colorbar()
       plt.show()
       input()
       '''                  
       return (GML, GMLX, GMLZ), RL, RLX, RLZ, SBR

def computeRayleighEquations(DIMS, REFS, depth, RLOPT, topdex, botdex, symmetricProfile):
       # Get options data
       width = RLOPT[1]
       applyTop = RLOPT[2]
       applyLateral = RLOPT[3]
       mu = RLOPT[4]
       
       # Get DIMS data
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       OPS = NX * NZ
       
       # Set up the Rayleigh field
       GML, RL, RLX, RLZ, SBR = computeRayleighField(DIMS, REFS, depth, width, \
                                                     applyTop, applyLateral, symmetricProfile)
       
       # Compute the diagonal for full Rayleigh field
       tempDiagonal = np.reshape(RL, (OPS,), order='F')
       # Compute the matrix operator
       RLM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       '''
       # Compute the diagonal for full Rayleigh field
       tempDiagonal = np.reshape(RLX, (OPS,), order='F')
       # Compute the matrix operator
       RLXM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       # Compute the diagonal for full Rayleigh field
       tempDiagonal = np.reshape(RLZ, (OPS,), order='F')
       # Compute the matrix operator
       RLZM = sps.spdiags(tempDiagonal, 0, OPS, OPS)
       '''
       # Store the diagonal blocks corresponding to Rayleigh damping terms
       ROPS = mu * np.array([RLM, RLM, RLM, RLM])
       
       return ROPS, RLM, GML
       
                            