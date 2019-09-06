#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:23:58 2019

@author: TempestGuerra
"""

import numpy as np
from numpy import multiply as mul
import math as mt

def computeGuellrichDomain2D(DIMS, REFS, hx, dhdx):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       
       # input REFS = [x, z, HFM, whf, CPM, wcp]
       x = REFS[0]
       z = REFS[1]
       
       # Compute the flat XZ mesh
       HTZL, dummy = np.meshgrid(hx,z);
       XL, ZL = np.meshgrid(x,z);
       
       # High Order Improved Guellrich coordinate 3 parameter function
       xi = 1.0 / ZH * ZL;
       ang = 0.5 * mt.pi * xi;
       AR = 1.0E-3;
       p = 20;
       q = 5;
       
       expdec = np.exp(-p/q * xi)
       cosvar = np.power(np.cos(ang), p)
       cosvard = np.power(np.cos(ang), p-1)
       fxi1 = mul(expdec, cosvar)
       fxi2 = AR * mul(xi, (1.0 - xi));
       fxi = np.add(fxi1, fxi2)
       
       dfdxi1 = -p/q * mul(expdec, cosvar)
       dfdxi2 = -(0.5 * p) * mt.pi * mul(mul(expdec, np.sin(ang)), cosvard)
       dfdxi3 = -AR * (1.0 - 2.0 * xi);
       dfdxi = np.add(np.add(dfdxi1, dfdxi2), dfdxi3)
       
       dzdh = fxi;
       dxidz = ZH + mul(HTZL, np.add(dfdxi, -fxi))
       sigma = ZH * np.power(dxidz, -1.0)
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       DZT = np.zeros((NZ,NX))
       
       for rr in range(NZ):
              ZTL[rr,:] = np.add(mul(dzdh[rr,:], hx), ZL[rr,:])
              DZT[rr,:] = mul(dzdh[rr,:], dhdx);
       
       return XL, ZTL, DZT, sigma, dzdh