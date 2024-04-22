#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:58:27 2019

@author: TempestGuerra
"""
import numpy as np
import scipy.interpolate as spi

def computeTemperatureProfileOnGrid(PHYS, REFS, Z_in, T_in, isSmooth, isUniform, RLOPT):
       
       # Get REFS data
       Z = REFS[1]
       
       TZ = np.zeros(Z.shape)
       DTDZ = np.zeros(Z.shape)
       
       if isUniform:
              # Loop over each column and evaluate termperature for uniform N
              T0 = T_in[0]
              A = PHYS[7]**2 / PHYS[0]
              C = PHYS[0] / PHYS[2]
              EXPF = np.exp(A * Z)
              TZ = T0 * EXPF + (C / A) * (1.0 - EXPF)
              DTDZ = (A * T0 - C) * EXPF
       else:
              if isSmooth:
                     T_int = spi.PchipInterpolator(Z_in, T_in)
              else:
                     T_int = spi.interp1d(Z_in, T_in, kind='linear')
                     
              # Loop over each column and evaluate interpolant
              TZ = T_int(Z)
              DTDZ = T_int(Z, nu=1)
                                                                                                   
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
       return TZ, DTDZ