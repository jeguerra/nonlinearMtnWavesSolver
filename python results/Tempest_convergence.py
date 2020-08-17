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
       HFM, DHFM = hcnw.hefuncm(NX, xint, True)

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

tdir = '/media/jeg/TransferDATA/Schar025m_tempest/' # home desktop
#tdir = '/Volumes/TransferDATA/Schar025m_tempest/' # Macbook Pro
hresl = [500, 250, 125, 62]
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
       
       # Read in the W data from Tempest
       WMOD = np.mean(m_fid.variables['W'][-20:-10,:,0,:], axis=0)
       
       HermCheb = True
       # Get the reference solution data
       if HermCheb:
              refname = tdir + 'restartDB_exactBCSchar_025m'
       else:
              refname = tdir + 'restartDB_exactBCSchar_025mP'
              
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
              DDX_1D, HF_TRANS, dummy = derv.computeHermiteFunctionDerivativeMatrix(DIMS)
       else:
              DDX_1D, HF_TRANS, dummy = derv.computeFourierDerivativeMatrix(DIMS)
       DDZ_1D, CH_TRANS, dummy = derv.computeChebyshevDerivativeMatrix(DIMS)
       
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

       #plt.figure()
       #plt.plot(REFS[0], WREF[0,:], 'k--', x, WREFint[0,:], 'b', x, WMOD[0,:], 'g')
       #plt.xlim(-15000.0, 15000.0)
       
       # Sample the interior flow
       lxi = int(0.25*len(x))
       lzi1 = int(0.23*len(z))
       lzi2 = int(0.35*len(z))
       
       xint = x[lxi-1:-lxi]
       zint = z[lzi1:-lzi2]
       # Make the differences
       WDIFF1 = WMOD - WREFint
       WDIFF2 = WMOD - WBK
       WDIFF3 = WREFint - WBK
       
       DOPS = len(x) * len(z)
       DOPSint = len(xint) * len(zint)
       
       # Compute norms (TEMPEST TO SPECTRAL REFERENCE)
       ndiff_wbc = np.linalg.norm(WDIFF1[0,:])
       ndiff_fld = np.linalg.norm(np.reshape(WDIFF1, (DOPS,), order='F'))
       ndiff_int = np.linalg.norm(np.reshape(WDIFF1[lzi1:-lzi2,lxi-1:-lxi], (DOPSint,), order='F'))
       
       nref_wbc = np.linalg.norm(WREFint[0,:])
       nref_fld = np.linalg.norm(np.reshape(WREFint, (DOPS,), order='F'))
       nref_int = np.linalg.norm(np.reshape(WREFint[lzi1:-lzi2,lxi-1:-lxi], (DOPSint,), order='F'))
       # Take the norm and print
       wbcerr1.append(ndiff_wbc / nref_wbc)
       wflerr1.append(ndiff_fld / nref_fld)
       wnterr1.append(ndiff_int / nref_int)
       
       # Compute norms (TEMPEST TO CLASSICAL REFERENCE)
       ndiff_wbc = np.linalg.norm(WDIFF2[0,:])
       ndiff_fld = np.linalg.norm(np.reshape(WDIFF2, (DOPS,), order='F'))
       ndiff_int = np.linalg.norm(np.reshape(WDIFF2[lzi1:-lzi2,lxi-1:-lxi], (DOPSint,), order='F'))
       
       nref_wbc = np.linalg.norm(WBK[0,:])
       nref_fld = np.linalg.norm(np.reshape(WBK, (DOPS,), order='F'))
       nref_int = np.linalg.norm(np.reshape(WBK[lzi1:-lzi2,lxi-1:-lxi], (DOPSint,), order='F'))
       # Take the norm and print
       wbcerr2.append(ndiff_wbc / nref_wbc)
       wflerr2.append(ndiff_fld / nref_fld)
       wnterr2.append(ndiff_int / nref_int)
       
       X, Z = np.meshgrid(1.0E-3 * x, 1.0E-3 * z)
       
       ccount = 40
       wfbound = 0.2
       # Plot the difference (Tempest to Spectral Reference)
       fig = plt.figure(figsize=(18.0, 3.0))
       plt.subplot(1,3,1)
       ccheck = plt.contourf(X, Z, WMOD, ccount, cmap=cm.seismic)
       plt.xlim(-30.0, 30.0)
       plt.clim(-wfbound, wfbound)
       norm = mpl.colors.Normalize(vmin=-wfbound, vmax=wfbound)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.title('Tempest Model W @ 20 Hours (m/s)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,3,2)
       ccheck = plt.contourf(X, Z, WREFint, ccount, cmap=cm.seismic)
       plt.xlim(-30.0, 30.0)
       plt.clim(-wfbound, wfbound)
       norm = mpl.colors.Normalize(vmin=-wfbound, vmax=wfbound)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.title('Steady Spectral Reference W (m/s)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,3,3)
       ccheck = plt.contourf(X, Z, WDIFF1, ccount, cmap=cm.seismic)
       plt.title('Difference (m/s): ' + str(rr) + ' (m)')
       wdbound = 0.025
       plt.clim(-wdbound, wdbound)
       norm = mpl.colors.Normalize(vmin=-wdbound, vmax=wdbound)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       '''
       # Plot the difference (Tempest to Classical Reference)
       fig = plt.figure(figsize=(18.0, 3.0))
       plt.subplot(1,3,1)
       ccheck = plt.contourf(WMOD, ccount, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       #plt.title('Tempest W (m/s): ' + str(rr) + ' (m)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       fig.colorbar(ccheck)
       plt.subplot(1,3,2)
       ccheck = plt.contourf(WBK, ccount, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       plt.title('Classical Reference W (m/s): ' + str(rr) + ' (m)')
       fig.colorbar(ccheck)
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,3,3)
       ccheck = plt.contourf(WDIFF2, ccount, cmap=cm.seismic)#, vmin=-0.1, vmax=0.1)
       plt.title('Difference (m/s): ' + str(rr) + ' (m)')
       plt.clim(wdmin, wdmax)
       norm = mpl.colors.Normalize(vmin=wdmin, vmax=wdmax)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       '''
       '''
       # Plot the difference (Spectral to Classical Reference)
       fig = plt.figure(figsize=(18.0, 3.0))
       plt.subplot(1,3,1)
       ccheck = plt.contourf(WREFint, ccount, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       plt.title('Spectral Reference W (m/s): ' + str(rr) + ' (m)')
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       fig.colorbar(ccheck)
       plt.subplot(1,3,2)
       ccheck = plt.contourf(WBK, ccount, cmap=cm.seismic)#, vmin=0.0, vmax=20.0)
       plt.title('Classical Reference W (m/s): ' + str(rr) + ' (m)')
       fig.colorbar(ccheck)
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       plt.subplot(1,3,3)
       ccheck = plt.contourf(WDIFF3, ccount, cmap=cm.seismic)#, vmin=-0.1, vmax=0.1)
       plt.title('Difference (m/s): ' + str(rr) + ' (m)')
       plt.clim(wdmin, wdmax)
       norm = mpl.colors.Normalize(vmin=wdmin, vmax=wdmax)
       plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic))
       plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
       '''
       #plt.xlim(-50.0, 50.0)
       #plt.ylim(0.0, 1.0E-3*DIMS[2])
       
print(wbcerr1)
print(wflerr1)
#%%       
fig = plt.figure(figsize=(5.0, 5.0))
plt.plot(hresl, wbcerr1, 'C0s-', hresl, wbcerr2, 'C1s-'); plt.ylim(1.0E-3, 1.0E-1)
plt.title('Terrain Boundary Response')
plt.xscale('linear'); plt.yscale('log')
plt.xlabel('Resolution (m)')
plt.ylabel('L-2 Norm W (m/s)')
plt.legend(('Spectral Reference','Classical Reference'), loc='lower right')
plt.grid(b=None, which='both', axis='both', color='k', linestyle='--', linewidth=0.5)
fig = plt.figure(figsize=(10.0, 5.0))
plt.subplot(1,2,1)
plt.plot(hresl, wnterr1, 'C0s-', hresl, wnterr2, 'C1s-'); plt.ylim(1.0E-3, 1.0)
plt.title('Interior Domain Response')
plt.xscale('linear'); plt.yscale('log')
plt.xlabel('Resolution (m)')
plt.ylabel('L-2 Norm W (m/s)')
plt.legend(('Spectral Reference','Classical Reference'), loc='lower right')
plt.grid(b=None, which='both', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.subplot(1,2,2)
plt.plot(hresl, wflerr1, 'C0s-'); plt.ylim(1.0E-3, 1.0E0)
plt.title('Entire Domain Response')
plt.xscale('linear'); plt.yscale('log')
plt.xlabel('Resolution (m)')
#plt.ylabel('L-2 Norm (m/s)')
#plt.legend(('Spectral Reference'), loc='lower right')
plt.grid(b=None, which='both', axis='both', color='k', linestyle='--', linewidth=0.5)
       
       