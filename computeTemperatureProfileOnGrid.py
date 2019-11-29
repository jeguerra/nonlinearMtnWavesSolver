#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:58:27 2019

@author: TempestGuerra
"""
import numpy as np

def computeTemperatureProfileOnGrid(Z_in, T_in, REFS, isSmooth):
       
       # Get REFS data
       z = REFS[1]
       
       if isSmooth:
              ZTP = Z_in[1] # tropopause height
              ZTM = Z_in[2] # top of stratospheric mixed layer
              ZH = Z_in[3] # top of the model atmosphere
              
              # 5th order polynomial fit coefficient matrix
              VandermondeM = np.array([[ZTP**2, ZTP**3, ZTP**4, ZTP**5], \
                                       [ZTM**2, ZTM**3, ZTM**4, ZTM**5], \
                                       [ZH**2, ZH**3, ZH**4, ZH**5], \
                                       [2*ZH, 3*ZH**2, 4*ZH**3, 5*ZH**4]])
              
              TS = T_in[0] # Surface temperature
              TTP = T_in[1] # Temperature at tropopause
              TTM = T_in[2] # Temperature at top of mixed layer
              TH = T_in[3] # Temperature at model top
              DTS = (TTP - TS) / (ZTP - Z_in[0])
              DTH = (TH - TTM) / (ZH - ZTM)
              
              # 5th order polynomial fit RHS
              VRHS = [TTP - TS - ZTP*DTS, \
                      TTM - TS - ZTM*DTS, \
                      TH - TS - ZH*DTS, \
                      DTH - DTS]
              
              coeffs = np.linalg.solve(VandermondeM, VRHS)
              
              # Evaluate the polynomial and derivative (lapse rates)
              TZ = TS + DTS * z + coeffs[0] * np.power(z,2) \
                                + coeffs[1] * np.power(z,3) \
                                + coeffs[2] * np.power(z,4) \
                                + coeffs[3] * np.power(z,5)
                                
              DTDZ = DTS + (2 * coeffs[0] * z) \
                         + (3 * coeffs[1] * np.power(z,2)) \
                         + (4 * coeffs[2] * np.power(z,3)) \
                         + (5 * coeffs[3] * np.power(z,4))
       else:
              # Get the 1D linear interpolation for this sounding
              TZ = np.interp(z, Z_in, T_in)
              
              # Get piece-wise derivatives
              DTDZ = np.zeros(len(z))
              # Loop over layers
              for pp in range(len(Z_in) - 1):
                     # Local lapse rate
                     LR = (T_in[pp+1] - T_in[pp]) / (Z_in[pp+1] - Z_in[pp])
                     # Loop over the layer
                     for kk in range(len(z)):
                            if (z[kk] >= Z_in[pp]) and (z[kk] <= Z_in[pp+1]):
                                   DTDZ[kk] = LR
       
       return TZ, DTDZ