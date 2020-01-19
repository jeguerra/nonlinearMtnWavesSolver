#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:21:27 2019

SUPPORTS AN ANALYTICAL PROFILE FOR WIND ONLY...

@author: TempestGuerra
"""

import numpy as np
import math as mt

def computeShearProfileOnGrid(REFS, JETOPS, P0, PZ, dlnPZdz, uniformWind, linearShear):
       
       # Get jet profile options
       U0 = JETOPS[0]
       uj = JETOPS[1]
       b = JETOPS[2]
       
       if uniformWind:
              UZ = U0 * np.ones(len(PZ))
              dUdz = np.zeros(len(PZ))  
       elif linearShear:
              zcoord = REFS[1]
              ZH = zcoord[len(zcoord)-1]
              DUZ = (10.0 / ZH**2) * np.power(zcoord, 2.0)
              UZ = DUZ + U0
              dUdz = (20.0 / ZH**2) * zcoord
       else:
              # Compute the normalized pressure coordinate (Ullrich, 2015)
              pcoord = 1.0 / P0 * PZ;
              lpcoord = np.log(pcoord);
              lpcoord2 = np.power(lpcoord, 2.0)
              
              # Compute the decay portion of the jet profile
              jetDecay = np.exp(-(1.0 / b**2.0 * lpcoord2));
              UZ = -uj * np.multiply(lpcoord, jetDecay) + U0;
           
              # Compute the shear
              temp = np.multiply(jetDecay, (1.0 - 2.0 / b**2 * lpcoord2))
              dUdz = -uj * temp * np.reciprocal(pcoord);
              dUdz *= (1.0 / P0);
              dUdz *= P0 * (pcoord * dlnPZdz);
       
       return UZ, dUdz
       
       