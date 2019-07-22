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
       
       return(TZ)