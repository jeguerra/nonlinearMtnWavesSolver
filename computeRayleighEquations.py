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

def computeRayleighField(DIMS, REFS, height, width, applyTop, applyLateral):
       
       # Get DIMS data
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       
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
       SBR = np.ones((NZ, NX))
       
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
                            RFX = 1.0 - (mt.sin(0.5 * mt.pi * dNormX))**RP
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
                            RFZ = 1.0 - (mt.sin(0.5 * mt.pi * dNormZ))**RP
                     else:
                            RFZ = 0.0
                     
                     # Set the field to max(lateral, top) to handle corners
                     RLX[ii,jj] = RFX
                     RLZ[ii,jj] = RFZ
                     RL[ii,jj] = np.amax([RFX, RFZ])
                     # Set the binary matrix
                     if RL[ii,jj] != 0.0:
                            SBR[ii,jj] = 0.0                            
       '''
       fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
       ax.plot_surface(X, Z, RL, cmap=cm.seismic,
                       linewidth=0, antialiased=False)
       plt.show()
       input('CHECK RAYLEIGH...')
       '''                     
       # Assemble the Grid Matching Layer field X and Z directions
       GML = np.ones((NZ, NX))
       GMLX = np.ones((NZ, NX))
       GMLZ = np.ones((NZ, NX))
       C1 = 0.02
       C2 = 10.0
       isStretchGML = True # True: trig GML to RHS, False, direct GML to state
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
                            
                            if isStretchGML:
                                   # Evaluate the GML factor
                                   #RFX = (mt.tan(0.5 * mt.pi * dNormX))**GP
                                   RFX = 2.0 * dNormX**2
                            else:
                                   # Evaluate buffer layer factor
                                   RFX = (1.0 - C1 * dNormX**2) * \
                                          (1.0 - (1.0 - mt.exp(C2 * dNormX**2)) / (1.0 - mt.exp(C2)))
                     else:
                            RFX = 0.0
                     if applyTop:
                            # In the top layer?
                            if ZRL >= dLayerZ[jj]:
                                   dNormZ = (ZRL - dLayerZ[jj]) / (ZH - height[jj])
                            else:
                                   dNormZ = 0.0
                                   
                            if isStretchGML:
                                   # Evaluate the strength of the field
                                   #RFZ = (mt.tan(0.5 * mt.pi * dNormZ))**GP
                                   RFZ = 2.0 * dNormZ**2
                            else:
                                   # Evaluate buffer layer factor
                                   RFZ = (1.0 - C1 * dNormZ**2) * \
                                          (1.0 - (1.0 - mt.exp(C2 * dNormZ**2)) / (1.0 - mt.exp(C2)))
                     else:
                            RFZ = 0.0
                     
                     if isStretchGML:
                            GMLX[ii,jj] = 1.0 / (1.0 + RFX)
                            GMLZ[ii,jj] = 1.0 / (1.0 + RFZ)
                            # Set the field to max(lateral, top) to handle corners
                            RFM = np.amax([RFX, RFZ])
                            GML[ii,jj] = 1.0 / (1.0 + RFM)
                     else:
                            GMLX[ii,jj] = RFX
                            GMLZ[ii,jj] = RFZ
                            # Set the field to max(lateral, top) to handle corners
                            GML[ii,jj] = np.amin([RFX, RFZ])
       '''
       plt.figure()
       plt.contourf(X, Z, GMLX, 101, cmap=cm.seismic)
       plt.colorbar()
       plt.show()
       input()
       '''                  
       return (GML, GMLX, GMLZ), RL, RLX, RLZ, SBR

def computeRayleighEquations(DIMS, REFS, depth, RLOPT, topdex, botdex):
       # Get options data
       width = RLOPT[1]
       applyTop = RLOPT[2]
       applyLateral = RLOPT[3]
       mu = RLOPT[4]
       
       # Get DIMS data
       NX = DIMS[3] + 1
       NZ = DIMS[4] + 1
       OPS = DIMS[5]
       
       # Set up the Rayleigh field
       GML, RL, RLX, RLZ, SBR = computeRayleighField(DIMS, REFS, depth, width, \
                                                     applyTop, applyLateral)
       
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
       
       return ROPS, RLM, GML, SBR
       
                            