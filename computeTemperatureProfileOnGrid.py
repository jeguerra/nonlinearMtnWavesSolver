#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:58:27 2019

@author: TempestGuerra
"""
import numpy as np

def computeTemperatureProfileOnGrid(Z_in, T_in, REFS):
       
       # Get REFS data
       z = REFS[1]
       
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