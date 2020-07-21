#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:47:04 2020

@author: jeg
"""

import shelve
import numpy as np
import math as mt
from matplotlib import cm
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import computeDerivativeMatrix as derv

def computeHorizontalInterp(NX, xint, FLD, HF_TRANS):
       import HerfunChebNodesWeights as hcnw
       
       # Compute coefficients for the variable field
       fcoeffs = np.matmul(HF_TRANS, FLD.T)
       
       xh, dummy = hcnw.hefunclb(NX)
       xint *= max(xh) / max(xint)
       
       # Compute backward transform to new grid
       HFM = hcnw.hefuncm(NX, xint, True)

       #plt.figure()
       #plt.plot(xh, HFM_native[0,:])
       #plt.figure()
       #plt.plot(xi, HFM[0,:])
       
       # Apply the backward transforms
       FLDI = np.matmul(HFM, fcoeffs)
       
       return FLDI.T

def computeColumnInterp(NX, NZ, NZI, ZTL, FLD, CH_TRANS):
       import HerfunChebNodesWeights as hcnw
                     
       # Interpolated field has a new size
       FLDI = np.zeros((NZI, NX+1))
       # Compute the new column reference grid (linear space)
       zint = np.linspace(-1.0, 1.0, num=NZI, endpoint=True)
       
       # Loop over each column
       for cc in range(NX):
              # Apply the forward transform at this column
              fcoeffs = CH_TRANS.dot(FLD[:,cc])
              
              # Get the Chebyshev matrix for this column
              CTM = hcnw.chebpolym(NZ-1, -zint)
              
              # Apply the interpolation
              temp = (CTM).dot(fcoeffs)
              #print(temp, cc)
              FLDI[:,cc] = np.ravel(temp)
              
       return FLDI

#tdir = '/media/jeg/TransferDATA/Schar025m_tempest/' # home desktop
tdir = '/Volumes/TransferDATA/Schar025m_tempest/' # Macbook Pro
hresl = [1000, 500, 250, 125]
# Loop over the 4 data files
for rr in hresl:
       fname = tdir + 'outNHGW_VO3_BSLN-SCHAR_H25m_' + str(rr) + 'm.0000-01-01-00000.nc'
       m_fid = Dataset(fname, 'r')
       # Read on grid data from Tempest
       x = m_fid.variables['lon'][:]
       z = m_fid.variables['lev'][:]
       hz = m_fid.variables['Zs']
       # Read in the W data from Tempest
       WMOD = m_fid.variables['W'][120,:,0,:]
       
       # Get the reference solution data
       refname = tdir + 'restartDB_exactBCSchar_025m'
       rdb = shelve.open(refname, flag='r')
       NX = rdb['NX']
       NZ = rdb['NZ']
       SOLT = rdb['SOLT']
       REFS = rdb['REFS']
       PHYS = rdb['PHYS']
       DIMS = rdb['DIMS']
       
       # Get the Hermite and Chebyshev transforms
       DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       DDZ_1D, CH_TRANS = derv.computeChebyshevDerivativeMatrix(DIMS)
       
       # Get the reference vertical velocity (on the native HermCheb grid)
       OPS = DIMS[5]
       fields = np.reshape(SOLT[:,0], (OPS, 4), order='F')
       WREF = np.reshape(fields[:,1], (NZ, NX+1), order='F')
       
       # Interpolate to the Tempest grid
       ZTL = REFS[5]
       NXI = len(x)
       NZI = len(z)
       WREFint = computeColumnInterp(NX, NZ, NZI, ZTL, WREF, CH_TRANS)
       WREFint = computeHorizontalInterp(NX, x, WREFint, HF_TRANS)
       
       # Plot the difference
       fig = plt.figure(figsize=(20.0, 6.0))
       plt.subplot(1,3,1)
       ccheck = plt.contourf(WMOD, 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       fig.colorbar(ccheck)
       plt.subplot(1,3,2)
       ccheck = plt.contourf(WREFint, 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       fig.colorbar(ccheck)
       plt.subplot(1,3,3)
       ccheck = plt.contourf(WMOD - WREFint, 201, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       fig.colorbar(ccheck)
       #plt.xlim(-50.0, 50.0)
       #plt.ylim(0.0, 1.0E-3*DIMS[2])
       
       
       