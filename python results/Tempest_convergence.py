#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:47:04 2020

@author: jeg
"""

import shelve
import scipy as scp
import numpy as np
import math as mt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import computeDerivativeMatrix as derv

def computeHorizontalInterpHermite(NX, xint, FLD, HF_TRANS):
       import HerfunChebNodesWeights as hcnw
       
       # Compute coefficients for the variable field
       fcoeffs = HF_TRANS.dot(FLD.T)
       
       xh, dummy = hcnw.hefunclb(NX)
       xint *= max(xh) / max(xint)
       
       # Compute backward transform to new grid
       HFM = hcnw.hefuncm(NX, xint, True)

       #plt.figure()
       #plt.plot(xh, HFM_native[0,:])
       #plt.figure()
       #plt.plot(xi, HFM[0,:])
       
       # Apply the backward transforms
       FLDI = HFM.dot(fcoeffs)
       
       return FLDI.T

def computeHorizontalInterpFourier(x0, xint, FLD, HF_TRANS):
       #'''
       NP = len(x0)
       LX = abs(max(x0) - min(x0))
       # Compute forward transform
       k = np.fft.fftfreq(NP)
       ks = 2.0*mt.pi/LX * k * NP
              
       # Compute interpolation by FFT
       HF = np.fft.fft(FLD, axis=1)
       # Compute the orthogonal projection to the xh grid
       NPI = len(xint)
       FIM = np.zeros((NPI, NP), dtype=complex)
       # Apply the spatial shift due to different grid
       dx = max(xint)
       HF *= np.exp(-1j * ks * dx / NP)
       # Compute the Fourier basis on the desired grid
       for cc in range(NP):
              arg = 1j * ks[cc] * xint#[0:-1]# * cc / NP
              FIM[:,cc] = 1.0 / NP * np.exp(arg)
                     
       # Compute the inverse Fourier interpolation
       FLDI = FIM.dot(HF.T)
       #'''
       return np.real(np.fft.fftshift(FLDI.T, axes=1))

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

#hdir = '/home/jeg/scratch/' # home desktop runs
hdir = '/media/jeg/TransferDATA/Schar025m_tempest/' # runs done on Farm
tdir = '/media/jeg/TransferDATA/Schar025m_tempest/Schar025m/' # home desktop
#tdir = '/Volumes/TransferDATA/Schar025m_tempest/' # Macbook Pro
hresl = [1000, 500, 250, 125, 62]
refName = 'restartDB_BC4'
#hresl = [250]
# Loop over the 4 data files
# Error norms between Tempest and Spectral Reference
wbcerr1 = []
wflerr1 = []
wnterr1 = []
# Error norms between Tempest and Classical Reference
wbcerr2 = []
wflerr2 = []
wnterr2 = []
for rr in hresl:
       fname = tdir + 'outNHGW_VO3_BSLN-SCHAR_H25m_' + str(rr) + 'm.0000-01-01-00000.nc'
       m_fid = Dataset(fname, 'r')
       # Read on grid data from Tempest
       x = m_fid.variables['lon'][:]
       z = m_fid.variables['lev'][:]
       hz = m_fid.variables['Zs']
       
       # Get model run at 20, 40, and 60 hours
       timeMin = 24 * 60 * m_fid.variables['time'][:]
       ts2get = [(np.argwhere(timeMin == 24 * 60))[0,0], \
                 (np.argwhere(timeMin == 48 * 60))[0,0], \
                 (np.argwhere(timeMin == 72 * 60))[0,0]]       
       
       # Read in the W data from Tempest
       WMOD = m_fid.variables['W'][ts2get,:,0,:]
       
       HermCheb = True
       # Get the reference solution data
       if HermCheb:
              refname = hdir + refName
       else:
              refname = hdir + 'restartDB_exactBCSchar_025mP'
              
       rdb = shelve.open(refname, flag='r')
       NX = rdb['NX']
       NZ = rdb['NZ']
       SOLT = rdb['SOLT']
       REFS = rdb['REFS']
       PHYS = rdb['PHYS']
       DIMS = rdb['DIMS']
       
       from BoussinesqSolSchar import ScharBoussinesqKlemp
       z *= max(REFS[1]) # convert to meters
       WBK, ETA, WFFT = ScharBoussinesqKlemp(PHYS, x, z)
       
       # Get the Hermite and Chebyshev transforms
       if HermCheb:
              DDX_1D, HF_TRANS = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       else:
              DDX_1D, HF_TRANS = derv.computeFourierDerivativeMatrix(DIMS)
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
       if HermCheb:
              WREFint = computeHorizontalInterpHermite(NX, np.array(x), WREFint, HF_TRANS)
       else:
              WREFint = computeHorizontalInterpFourier(np.array(REFS[0]), np.array(x), WREFint, HF_TRANS)
       
       # Sample the interior flow
       xintDex = (np.argwhere(abs(x) <= 15000.0))[:,0]
       zintDex1 = set((np.argwhere(z <= 15000.0))[:,0])
       zintDex2 = set((np.argwhere(z >= 5000.0))[:,0])
       zintDex = sorted(zintDex1.intersection(zintDex2))
       intDex = np.ix_(zintDex, xintDex)
       
       # Make the differences
       WDIFF1 = WMOD - np.expand_dims(WREFint, axis=0)
       WDIFF2 = WMOD - np.expand_dims(WBK, axis=0)
       WDIFF3 = WREFint - WBK
       
       DOPS = len(x) * len(z)
       DOPSint = len(xintDex) * len(zintDex)
       
       # Compute norms (TEMPEST TO SPECTRAL REFERENCE)
       ndiff_wbc = np.linalg.norm(WDIFF1[:,0,:], axis=1)
       ndiff_fld = np.linalg.norm(np.reshape(WDIFF1, (len(ts2get),DOPS), order='F'), axis=1)
       ndiff_int = np.linalg.norm(np.reshape(WDIFF1[:,intDex[0],intDex[1]], (len(ts2get),DOPSint), order='F'), axis=1)
       
       nref_wbc = np.linalg.norm(WREFint[0,:])
       nref_fld = np.linalg.norm(np.reshape(WREFint, (DOPS,), order='F'))
       nref_int = np.linalg.norm(np.reshape(WREFint[intDex[0],intDex[1]], (DOPSint,), order='F'))
       # Take the norm and print
       wbcerr1.append(ndiff_wbc / nref_wbc)
       wflerr1.append(ndiff_fld / nref_fld)
       wnterr1.append(ndiff_int / nref_int)
       
       # Compute norms (TEMPEST TO CLASSICAL REFERENCE)
       ndiff_wbc = np.linalg.norm(WDIFF2[:,0,:], axis=1)
       ndiff_fld = np.linalg.norm(np.reshape(WDIFF2, (len(ts2get),DOPS), order='F'), axis=1)
       ndiff_int = np.linalg.norm(np.reshape(WDIFF2[:,intDex[0],intDex[1]], (len(ts2get),DOPSint), order='F'), axis=1)
       
       nref_wbc = np.linalg.norm(WBK[0,:])
       nref_fld = np.linalg.norm(np.reshape(WBK, (DOPS,), order='F'))
       nref_int = np.linalg.norm(np.reshape(WBK[intDex[0],intDex[1]], (DOPSint,), order='F'))
       # Take the norm and print
       wbcerr2.append(ndiff_wbc / nref_wbc)
       wflerr2.append(ndiff_fld / nref_fld)
       wnterr2.append(ndiff_int / nref_int)
       
       X, Z = np.meshgrid(1.0E-3 * x, 1.0E-3 * z)
       
       ccount = 40
       wfbound = 0.2
       # Plot the difference (Tempest to Spectral Reference)
       fig = plt.figure(figsize=(12.0, 4.0))
       plt.subplot(1,2,1)
       ccheck = plt.contourf(X, Z, WMOD[-1,:,:], ccount, cmap=cm.seismic)
       plt.contour(X, Z, WREFint, ccount, colors='k', linewidths=0.5)
       plt.xlim(-20.0, 20.0)
       plt.clim(-wfbound, wfbound)
       plt.xlabel('X (km)')
       plt.ylabel('Z (km)')
       norm = mpl.colors.Normalize(vmin=-wfbound, vmax=wfbound)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.title('Tempest vs. Reference W (m/s)')
       #plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,2,2)
       ccheck = plt.contourf(X, Z, WDIFF1[2,:,:], ccount, cmap=cm.seismic)
       plt.title('Difference W (m/s) @ 72Hr: ' + str(rr) + ' (m)')
       wdbound = 0.02
       plt.clim(-wdbound, wdbound)
       plt.xlabel('X (km)')
       plt.ylabel('Z (km)')
       norm = mpl.colors.Normalize(vmin=-wdbound, vmax=wdbound)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       
wbcerr1 = np.array(wbcerr1)
wbcerr2 = np.array(wbcerr2)
wflerr1 = np.array(wflerr1)
wflerr2 = np.array(wflerr2)
wnterr1 = np.array(wnterr1)
wnterr2 = np.array(wnterr2)
print(wbcerr1)
print(wflerr1)
#%%  
convRate = np.power(1.0E-3 * np.array(hresl), 1.5)
convRate *= 1.0 / np.amax(convRate)
     
fig = plt.figure(figsize=(5.0, 5.0))
plt.plot(hresl, wbcerr2[:,2], 's-', hresl, wbcerr1[:,2], 's-'); plt.ylim(1.0E-3, 1.0E0)
plt.plot(hresl, np.amax(wbcerr1[:,2]) * convRate, 'k--')
plt.title('Terrain Boundary Response')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Resolution (m)')
plt.ylabel('L-2 Norm W (m/s)')
plt.legend(('Classical Reference','Spectral 72Hr'), loc='lower right')
plt.grid(b=None, which='both', axis='both', color='k', linestyle='--', linewidth=0.5)
fig = plt.figure(figsize=(10.0, 5.0))
plt.subplot(1,2,1)
plt.plot(hresl, wnterr2[:,2], 'ks-', hresl, wnterr1, 's-'); plt.ylim(1.0E-3, 5.0E-1)
plt.plot(hresl, np.amax(wnterr1) * convRate, 'k--')
plt.title('Interior Domain Response')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Resolution (m)')
plt.ylabel('L-2 Norm W (m/s)')
plt.legend(('Classical Reference','Spectral 24Hr','Spectral 48Hr','Spectral 72Hr'), loc='lower right')
plt.grid(b=None, which='both', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.subplot(1,2,2)
plt.plot(hresl, wflerr1, 's-'); plt.ylim(1.0E-3, 5.0E-1)
plt.plot(hresl, np.amax(wflerr1) * convRate, 'k--')
plt.title('Entire Domain Response')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Resolution (m)')
#plt.ylabel('L-2 Norm (m/s)')
plt.legend(('Spectral 24Hr','Spectral 48Hr','Spectral 72Hr'), loc='lower right')
plt.grid(b=None, which='both', axis='both', color='k', linestyle='--', linewidth=0.5)
       
       