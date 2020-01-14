#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:58:27 2019

@author: TempestGuerra
"""
import numpy as np

def computeTemperatureProfileOnGrid(PHYS, REFS, Z_in, T_in, isSmooth, isUniform):
       
       # Get REFS data
       z = REFS[1]
       ZTL = REFS[5]
       NC = len(ZTL[0,:])
       
       if isUniform:
              # Loop over each column and evaluate termperature for uniform N
              TZ = np.zeros(ZTL.shape)
              DTDZ = np.zeros(ZTL.shape)
              T0 = T_in[0]
              C0 = (PHYS[4] * PHYS[0]**2) / (PHYS[3] * PHYS[7]**2)
              C1 = PHYS[7]**2 / PHYS[0]
              for cc in range(NC):
                     zcol = ZTL[:,cc]
                     TZ[:,cc] = (T0 - C0) * np.exp(C1*zcol) + C0
                     DTDZ[:,cc] = C1 * (T0 - C0) * np.exp(C1*zcol)
       else:
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
                     
                     # Loop over each column and evaluate interpolant
                     TZ = np.zeros(ZTL.shape)
                     DTDZ = np.zeros(ZTL.shape)
                     for cc in range(NC):
                            zcol = ZTL[:,cc]
                            # Evaluate the polynomial and derivative (lapse rates)
                            TZ[:,cc] = TS + DTS * zcol + coeffs[0] * np.power(zcol,2) \
                                              + coeffs[1] * np.power(zcol,3) \
                                              + coeffs[2] * np.power(zcol,4) \
                                              + coeffs[3] * np.power(zcol,5)
                                              
                            DTDZ[:,cc] = DTS + (2 * coeffs[0] * zcol) \
                                       + (3 * coeffs[1] * np.power(zcol,2)) \
                                       + (4 * coeffs[2] * np.power(zcol,3)) \
                                       + (5 * coeffs[3] * np.power(zcol,4))
              else:
                     # Loop over each column and evaluate interpolant
                     TZ = np.zeros(ZTL.shape)
                     DTDZ = np.zeros(ZTL.shape)
                     for cc in range(NC):
                            zcol = ZTL[:,cc]
                            # Get the 1D linear interpolation for this sounding
                            TZ[:,cc] = np.interp(zcol, Z_in, T_in)
                            # Get piece-wise derivatives, loop over layers
                            for pp in range(len(Z_in) - 1):
                                   # Local lapse rate
                                   LR = (T_in[pp+1] - T_in[pp]) / (Z_in[pp+1] - Z_in[pp])
                                   # Loop over the layer
                                   for kk in range(len(zcol)):
                                          if (z[kk] >= Z_in[pp]) and (z[kk] <= Z_in[pp+1]):
                                                 DTDZ[kk,cc] = LR
       
       return TZ, DTDZ