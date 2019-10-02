#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:20:43 2019

@author: TempestGuerra
"""

import numpy as np
from computeColumnInterp import computeColumnInterp
from computeHorizontalInterp import computeHorizontalInterp

def computeInterpolatedFields(DIMS, ZTL, sol, NX, NZ, NXI, NZI, udex, wdex, pdex, tdex, CH_TRANS, HF_TRANS):
       
       #% Get the fields in physical space
       uxz = np.reshape(sol[udex], (NZ, NX+1), order='F')
       wxz = np.reshape(sol[wdex], (NZ, NX+1), order='F')
       pxz = np.reshape(sol[pdex], (NZ, NX+1), order='F')
       txz = np.reshape(sol[tdex], (NZ, NX+1), order='F')
       print('Recover solution on native grid: DONE!')
       
       #% Interpolate columns to a finer grid for plotting
       uxzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, uxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       wxzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, wxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       pxzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, pxz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       txzint = computeColumnInterp(DIMS, None, None, NZI, ZTL, txz, CH_TRANS, 'TerrainFollowingCheb2Lin')
       print('Interpolate columns to finer grid: DONE!')
       
       #% Interpolate rows to a finer grid for plotting
       uxzint = computeHorizontalInterp(DIMS, NXI, uxzint, HF_TRANS)
       wxzint = computeHorizontalInterp(DIMS, NXI, wxzint, HF_TRANS)
       pxzint = computeHorizontalInterp(DIMS, NXI, pxzint, HF_TRANS)
       txzint = computeHorizontalInterp(DIMS, NXI, txzint, HF_TRANS)
       print('Interpolate columns to finer grid: DONE!')
       
       native = [uxz, wxz, pxz, txz]
       interp = [uxzint, wxzint, pxzint, txzint]
       
       return native, interp