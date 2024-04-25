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
def computeRayleighField(DIMS, X, Z, height, width, applyTop, applyLateral):
       
       # Get DIMS data
       L1 = DIMS[0]
       L2 = DIMS[1]
       ZH = DIMS[2]
       
       RP = 2.0
       T1 = 0.25
       S1 = 0.25 / (1.0 - T1)
       C1 = 20.0
       
       # Get domain data
       NX = X.shape[1]
       NZ = Z.shape[0]
       
       RF = 0.1
       FM = 1.0
       
       # Set the layer bounds
       width2 = width * (1.0 - RF * np.sin(FM * mt.pi / ZH * Z[:,-1]))
       dLayerR = L2 - width2
       
       width1 = width * (1.0 - RF * np.sin(FM * mt.pi / ZH * Z[:,0]))
       dLayerL = L1 + width1
       
       shift = 0.5 * abs(L2 - L1)
       depth = ZH - height
       depth *= (1.0 + RF * np.sin(FM * mt.pi / (L2 - L1) * (X[-1,:] - shift)))
       dLayerZ = ZH - depth       
       
       # Assemble the Rayleigh field
       RL_inl = np.zeros((NZ, NX))
       RL_inr = np.zeros((NZ, NX))
       RL_top = np.zeros((NZ, NX))
       RL_all = np.zeros((NZ, NX))
       
       for ii in range(0,NZ):
              for jj in range(0,NX):
                     # Get this X location
                     XRL = X[ii,jj]
                     ZRL = Z[ii,jj]
                     
                     if applyLateral:
                            RFX1 = 0.0
                            RFX2 = 0.0
                            # Left layer or right layer or not? [1 0]
                            if XRL > dLayerR[ii]:
                                   dNormX = 2.0 * S1 * mt.pi * ((L2 - XRL) / width2[ii] - T1)
                                   RFX1 = 0.0
                                   #RFX2 = 1.0 / (1.0 + (mt.tan(dNormX))**RP)
                                   #'''
                                   if dNormX > 0.0:
                                          RFX2 = 1.0 / (1.0 + (mt.tan(dNormX))**RP)
                                   elif dNormX <= 0.0:
                                          RFX2 = 1.0
                                   #'''
                            elif XRL < dLayerL[ii]:
                                   dNormX = 2.0 * S1 * mt.pi * ((XRL - L1) / width1[ii] - T1)
                                   #RFX1 = 1.0 / (1.0 + (mt.tan(dNormX))**RP)
                                   RFX2 = 0.0
                                   #'''
                                   if dNormX > 0.0:
                                          RFX1 = 1.0 / (1.0 + (mt.tan(dNormX))**RP)
                                   elif dNormX <= 0.0:
                                          RFX1 = 1.0
                                   #'''
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
                                   dNormZ = 2.0 * S1 * mt.pi * ((ZH - ZRL) / depth[jj] - T1)
                                   
                                   if dNormZ > 0.0:
                                          RFZ = 1.0 / (1.0 + (mt.tan(dNormZ))**RP)
                                   elif dNormZ <= 0.0:
                                          RFZ = 1.0                                          
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
                            
                     # Absorption to the inflow lateral boundary
                     RL_inl[ii,jj] = RFX1
                     # Absorption to the outflow lateral boundary
                     RL_inr[ii,jj] = RFX2
                     # Absorption to the model top boundary
                     RL_top[ii,jj] = RFZ
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
                                   dNormX = (XRL - dLayerR[ii]) / width2[ii]
                                   GFX2 = sigma_func(dNormX)
                            elif XRL < dLayerL[ii]:
                                   GFX2 = 0.0
                                   dNormX = (dLayerL[ii] - XRL) / width1[ii]
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
              ax.plot_surface(X, Z, RL_all, cmap=cm.jet,
                              linewidth=0, antialiased=False)
              plt.show()
              input('CHECK BOUNDARY LAYERS...')
       
       return (GML, GMLX2, GMLZ), (RL_inl, RL_inr, RL_top, RL_all)

def computeRayleighEquations(DIMS, X, Z, depth, RLOPT):
       
       # Get options data
       width = RLOPT[1]
       applyTop = RLOPT[2]
       applyLateral = RLOPT[3]
       
       # Set up the Rayleigh field
       GL, RL = computeRayleighField(DIMS, X, Z, depth, width, \
                                                     applyTop, applyLateral)
       
       # Compute the diagonal for full Rayleigh field as matrices    
       OPS = RL[0].shape[0] * RL[0].shape[1]
       
       RLML = np.reshape(RL[0], (OPS,), order='F')
       RLMR = np.reshape(RL[1], (OPS,), order='F')
       RLMT = np.reshape(RL[2], (OPS,), order='F')
       RLMA = np.reshape(RL[3], (OPS,1), order='F')
       
       GLM = sps.spdiags(np.reshape(GL[0], (OPS,), order='F'), 0, OPS, OPS)
       GLMX = sps.spdiags(np.reshape(GL[1], (OPS,), order='F'), 0, OPS, OPS)
       GLMZ = sps.spdiags(np.reshape(GL[2], (OPS,), order='F'), 0, OPS, OPS)
       
       return (RLML, RLMR, RLMT, RLMA), (GLM, GLMX, GLMZ)
       
                            
