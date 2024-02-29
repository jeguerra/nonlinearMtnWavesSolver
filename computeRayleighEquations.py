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
checkPlots = False

def computeRayleighField(DIMS, REFS, height, width, applyTop, applyLateral):
       
       # Get DIMS data
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       
       RP = 4.0
       T1 = 1.0 / 2.0
       W1 = 1.0 / 3.0
       X1 = 2.0 / 1.0
       C1 = 20.0
       
       # Get REFS data
       X = REFS[4]
       Z = REFS[5]
       NX = X.shape[1]
       NZ = Z.shape[0]
       
       rd = (ZH - height[0])
       pert_width = 0.01 * width
       pert_depth = 0.01 * rd
       
       # Set the layer bounds
       width += pert_width * np.sin(4.0 * mt.pi / width * Z[:,-1])
       dLayerR = L2 - width
       
       width += pert_width * np.sin(4.0 * mt.pi / width * Z[:,0])
       dLayerL = L1 + width
       
       dLayerZ = height + pert_depth * np.sin(2.0 * mt.pi / rd * X[-1,:])
       depth = ZH - dLayerZ       
       
       # Assemble the Rayleigh field
       RL_in = np.zeros((NZ, NX))
       RL_out = np.zeros((NZ, NX))
       RL_all = np.zeros((NZ, NX))
       RLX1 = np.zeros((NZ, NX))
       RLX2 = np.zeros((NZ, NX))
       
       for ii in range(0,NZ):
              for jj in range(0,NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     if applyLateral:
                            # Left layer or right layer or not? [1 0]
                            if XRL > dLayerR[ii]:
                                   dNormX = X1 * (L2 - XRL) / width[ii] - 0.5
                                   RFX1 = 0.0
                                   if dNormX > 0.0:
                                          RFX2 = 1.0 / (1.0 + T1 * (mt.tan(W1 * mt.pi * dNormX))**RP)
                                   elif dNormX <= 0.0:
                                          RFX2 = 1.0
                                          
                                   RFX2 *= 1.0 - (2.0 * (L2 - XRL) / width[ii] - 1.0)**C1
                            elif XRL < dLayerL[ii]:
                                   dNormX = X1 * (XRL - L1) / width[ii] - 0.5
                                   RFX2 = 0.0
                                   if dNormX > 0.0:
                                          RFX1 = 1.0 / (1.0 + T1 * (mt.tan(W1 * mt.pi * dNormX))**RP)
                                   elif dNormX <= 0.0:
                                          RFX1 = 1.0
                                          
                                   RFX1 *= 1.0 - (2.0 * (XRL - L1) / width[ii] - 1.0)**C1
                            else:
                                   dNormX = 1.0
                                   RFX1 = 0.0
                                   RFX2 = 0.0
                     else:
                            RFX1 = 0.0
                            RFX2 = 0.0
                            
                     if applyTop:
                            # In the top layer?
                            if ZRL > dLayerZ[jj]:
                                   # This maps [depth ZH] to [1 0]
                                   dNormZ = X1 * (ZH - ZRL) / depth[jj] - 0.5
                                   
                                   if dNormZ > 0.0:
                                          RFZ = 1.0 / (1.0 + T1 * (mt.tan(W1 * mt.pi * dNormZ))**RP)
                                   elif dNormZ <= 0.0:
                                          RFZ = 1.0
                                          
                                   RFZ *= 1.0 - (2.0 * (ZH - ZRL) / depth[jj] - 1.0)**C1
                            else:
                                   dNormZ = 1.0
                                   RFZ = 0.0
                     else:
                            RFZ = 0.0
                            
                     if RFX1 < 0.0:
                            RFX1 = 0.0
                            
                     if RFX2 < 0.0:
                            RFX2 = 0.0
                            
                     if RFZ < 0.0:
                            RFZ = 0.0
                            
                     # Set the field to max(lateral, top) to handle corners
                     RLX1[ii,jj] = RFX1
                     RLX2[ii,jj] = RFX2
                     # Absorption to the inflow boundary
                     RL_in[ii,jj] = RLX1[ii,jj]
                     # Absorption to the outflow boundaries
                     RL_out[ii,jj] = np.amax([RFX2, RFZ])
                     # Complete absorption frame
                     RL_all[ii,jj] = np.amax([RFX1, RFX2, RFZ])
                            
       # Assemble the Grid Matching Layer field X and Z directions
       GML = np.ones((NZ, NX))
       GMLX1 = np.ones((NZ, NX))
       GMLX2 = np.ones((NZ, NX))
       GMLZ = np.ones((NZ, NX))
       
       def sigma_func(x):
              eps = 0.0
              p = 4.0
              q = 4.0
              sf = (1.0 - eps) * np.power(1.0 - np.power(1.0 - x, p), q)
              
              return sf
       
       for ii in range(0,NZ):
              for jj in range(0,NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     if applyLateral:
                            # Left layer or right layer or not? [0 1]
                            if XRL > dLayerR[ii]:
                                   GFX1 = 0.0
                                   dNormX = (XRL - dLayerR[ii]) / width[ii]
                                   GFX2 = sigma_func(dNormX)
                            elif XRL < dLayerL[ii]:
                                   GFX2 = 0.0
                                   dNormX = (dLayerL[ii] - XRL) / width[ii]
                                   GFX1 = sigma_func(dNormX)
                            else:
                                   dNormX = 0.0
                                   GFX1 = 0.0
                                   GFX2 = 0.0
                     else:
                            GFX1 = 0.0
                            GFX2 = 0.0
                            
                     if applyTop:
                            # In the top layer?
                            if ZRL > dLayerZ[jj]:
                                   dNormZ = (ZRL - dLayerZ[jj]) / depth[jj]
                                   GFZ = sigma_func(dNormZ)
                            else:
                                   GFZ = 0.0
                     else:
                            GFZ = 0.0
                     
                     GMLX2[ii,jj] = (1.0 - GFX2)
                     GMLZ[ii,jj] = (1.0 - GFZ)
                     # Set the field to max(lateral, top) to handle corners
                     GML[ii,jj] = np.amin([GMLX2[ii,jj], GMLZ[ii,jj]])

       
       if checkPlots:
       
              from matplotlib import cm
              import matplotlib.pyplot as plt
              fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
              ax.plot_surface(X, Z, RLX1, cmap=cm.jet,
                              linewidth=0, antialiased=False)
       
              fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
              ax.plot_surface(X, Z, GMLZ, cmap=cm.jet,
                              linewidth=0, antialiased=False)
              plt.show()
              input('CHECK BOUNDARY LAYERS...')
       
       return (GML, GMLX2, GMLZ), (RL_in, RL_out, RL_all)

def computeRayleighEquations(DIMS, REFS, depth, RLOPT):
       
       # Get options data
       width = RLOPT[1]
       applyTop = RLOPT[2]
       applyLateral = RLOPT[3]
       mu = RLOPT[4]
       
       # Set up the Rayleigh field
       GL, RL = computeRayleighField(DIMS, REFS, depth, width, \
                                                     applyTop, applyLateral)
       
       # Compute the diagonal for full Rayleigh field as matrices    
       OPS = RL[0].shape[0] * RL[0].shape[1]
       
       RLMI = np.reshape(RL[0], (OPS,1), order='F')
       RLMO = np.reshape(RL[1], (OPS,), order='F')
       RLMA = np.reshape(RL[2], (OPS,), order='F')
       
       GLM = sps.spdiags(np.reshape(GL[0], (OPS,), order='F'), 0, OPS, OPS)
       GLMX = sps.spdiags(np.reshape(GL[1], (OPS,), order='F'), 0, OPS, OPS)
       GLMZ = sps.spdiags(np.reshape(GL[2], (OPS,), order='F'), 0, OPS, OPS)

       # Store the diagonal blocks corresponding to Rayleigh damping terms
       RLM = sps.spdiags(np.reshape(RL[0], (OPS,), order='F'), 0, OPS, OPS)
       ROPS = mu * np.array([RLM, RLM, RLM, RLM])
       
       # Get the indices for the layer regions
       tempDiagonal = np.reshape(RL[0], (OPS,), order='F')
       ldex = np.nonzero(tempDiagonal)
       
       return ROPS, (RLMI, RLMO, RLMA), (GLM, GLMX, GLMZ), ldex[0]
       
                            