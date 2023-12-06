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

def computeRayleighField(DIMS, REFS, height, width, applyTop, applyLateral):
       
       # Get DIMS data
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       
       RP = 4.0
       C2 = 5.0
       W1 = 1.0 / 3.0
       X1 = 2.0 / 1.0
       
       # Get REFS data
       X = REFS[4]
       Z = REFS[5]
       NX = X.shape[1]
       NZ = Z.shape[0]
       
       rd = (ZH - height[0])
       pert_width = 0.1 * width
       pert_depth = 0.1 * rd
       
       # Set the layer bounds
       width += pert_width * np.sin(6.0 * mt.pi / width * Z[:,-1])
       dLayerR = (L2 - width) 
       
       width += pert_width * np.sin(6.0 * mt.pi / width * Z[:,0])
       dLayerL = L1 + width
       
       dLayerZ = height + pert_depth * np.sin(2.0 * mt.pi / rd * X[-1,:])
       depth = ZH - dLayerZ       
       
       # Assemble the Rayleigh field
       RL = np.zeros((NZ, NX))
       RLX = np.zeros((NZ, NX))
       RLZ = np.zeros((NZ, NX))
       
       for ii in range(0,NZ):
              for jj in range(0,NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     if applyLateral:
                            # Left layer or right layer or not? [1 0]
                            if XRL > dLayerR[ii]:
                                   dNormX = X1 * (L2 - XRL) / width[ii] - 0.5
                                   
                                   if dNormX > 0.0:
                                          RFX = 1.0 / (1.0 + (mt.tan(W1 * mt.pi * dNormX))**RP)
                                   elif dNormX <= 0.0:
                                          RFX = 1.0
                            elif XRL < dLayerL[ii]:
                                   dNormX = X1 * (XRL - L1) / width[ii] - 0.5
                                   
                                   if dNormX > 0.0:
                                          RFX = 1.0 / (1.0 + (mt.tan(W1 * mt.pi * dNormX))**RP)
                                   elif dNormX <= 0.0:
                                          RFX = 1.0
                            else:
                                   dNormX = 1.0
                                   RFX = 0.0
                     else:
                            RFX = 0.0
                     if applyTop:
                            # In the top layer?
                            if ZRL > dLayerZ[jj]:
                                   # This maps [depth ZH] to [1 0]
                                   dNormZ = X1 * (ZH - ZRL) / depth[jj] - 0.5
                                   
                                   if dNormZ > 0.0:
                                          RFZ = 1.0 / (1.0 + (mt.tan(W1 * mt.pi * dNormZ))**RP)
                                   elif dNormZ <= 0.0:
                                          RFZ = 1.0                
                            else:
                                   dNormZ = 1.0
                                   RFZ = 0.0
                     else:
                            RFZ = 0.0
                            
                     if RFX < 0.0:
                            RFX = 0.0
                            
                     if RFZ < 0.0:
                            RFZ = 0.0
                     
                     # Set the field to max(lateral, top) to handle corners
                     RLX[ii,jj] = RFX
                     RLZ[ii,jj] = RFZ
                     RL[ii,jj] = np.amax([RFX, RFZ]) #np.amax([0.5 * (RFX + RFZ), RFX, RFZ])
       
       '''
       from matplotlib import cm
       import matplotlib.pyplot as plt
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
       '''
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
       '''
       plt.figure()
       plt.contourf(X, Z, GMLX, 101, cmap=cm.seismic)
       plt.colorbar()
       plt.show()
       input()
       '''                  
       return (GML, GMLX, GMLZ), RL, RLX, RLZ

def computeRayleighEquations(DIMS, REFS, depth, RLOPT):
       
       # Get options data
       width = RLOPT[1]
       applyTop = RLOPT[2]
       applyLateral = RLOPT[3]
       mu = RLOPT[4]
       
       # Set up the Rayleigh field
       GML, RL, RLX, RLZ = computeRayleighField(DIMS, REFS, depth, width, \
                                                     applyTop, applyLateral)
       
       # Compute the diagonal for full Rayleigh field as matrices    
       OPS = RL.shape[0] * RL.shape[1]
       RLM = sps.spdiags(np.reshape(RL, (OPS,), order='F'), 0, OPS, OPS)
       RLMX = sps.spdiags(np.reshape(RLX, (OPS,), order='F'), 0, OPS, OPS)
       RLMZ = sps.spdiags(np.reshape(RLZ, (OPS,), order='F'), 0, OPS, OPS)

       # Store the diagonal blocks corresponding to Rayleigh damping terms
       ROPS = mu * np.array([RLM, RLM, RLM, RLM])
       
       # Get the indices for the layer regions
       tempDiagonal = np.reshape(RL, (OPS,), order='F')
       ldex = np.nonzero(tempDiagonal)
       
       return ROPS, (RLM, RLMX, RLMZ), GML, ldex[0]
       
                            