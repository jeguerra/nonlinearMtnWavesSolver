#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:58:27 2019

@author: TempestGuerra
"""
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi

def computeTemperatureProfileOnGrid(PHYS, REFS, Z_in, T_in, isSmooth, isUniform):
       
       # Get REFS data
       Z = REFS[1]
       
       TZ = np.zeros(Z.shape)
       DTDZ = np.zeros(Z.shape)
       
       DZ_AVG = np.mean(np.diff(Z))
       LT = 8 * DZ_AVG # transition length scale in meters
       
       if isUniform:
              # Loop over each column and evaluate termperature for uniform N
              T0 = T_in[0]
              A = PHYS[7]**2 / PHYS[0]
              C = PHYS[0] / PHYS[2]
              EXPF = np.exp(A * Z)
              TZS = T0 * EXPF + (C / A) * (1.0 - EXPF)
              DTDZ = (A * T0 - C) * EXPF
       else:
              T_int = spi.interp1d(Z_in, T_in, kind='linear')
              TZ = T_int(Z)
              DTDZ = np.gradient(TZ, Z, edge_order=1)
              TZS = np.copy(TZ)
              
              # Apply a Continuous Hermite Cubic Spline to layer interfaces
              if isSmooth:                     
                     
                     # Temperature changes one K over this length scale
                     for kk in np.arange(len(Z_in)):
                            
                            if kk >= 1 and kk < len(Z_in) - 1:
                                   
                                   ZI = Z_in[kk]
                                   
                                   # Get indices for the transition layer
                                   slayr = np.nonzero((Z >= ZI-LT) & \
                                                      (Z <= ZI+LT))
                                   
                                   # Index the end points
                                   sdex = slayr[0]
                                   edex = np.array((sdex[0],sdex[-1]))
                                   TI_int = spi.CubicHermiteSpline(Z[edex], TZ[edex], DTDZ[edex])
                                   
                                   TZS[sdex] = TI_int(Z[sdex])
                     
              # Loop over each column and evaluate interpolant
              DTDZ = np.gradient(TZS, Z, edge_order=1)
              '''
              plt.plot(Z,TZ,Z,TZS,linewidth=4.0)
              plt.figure()
              plt.plot(Z,DTDZ,linewidth=4.0)
              plt.show()
              input('TEMPERATURE SOUNDING CHECK.')
              '''
              '''
              # Make profile dry adiabatic near the top boundary (unsupportive of waves)
              if not isStatic:
                     C = PHYS[0] / PHYS[2]
                     for cc in range(NC):
                            zcol = ZTL[:,cc]
                            zrl = np.argwhere(zcol > (zcol[-1] - 0.5 * RLOPT[0]))
                            zrcol = ZTL[zrl,cc]
                            TZ[zrl,cc] = TZ[zrl[0],cc] - C * (zrcol - zrcol[0])
                            DTDZ[zrl,cc] = -C
              '''
       return TZS, DTDZ