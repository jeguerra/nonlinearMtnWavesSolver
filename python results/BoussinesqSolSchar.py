#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:25:16 2020

@author: jeg
"""

import numpy as np
import math as mt
from matplotlib import cm
import matplotlib.pyplot as plt

def ScharBoussinesqKlemp(PHYS, xout, zout):
       gc = PHYS[0]
       p0 = PHYS[1]
       cp = PHYS[2]
       Rd = PHYS[3]
       cv = cp - Rd
       
       T0 = 280.0
       U = 10.0
       N = 0.01
       ac = 5000.0
       hc = 25.0
       lc = 4000.0
       
       # Make the forward transform modal grid
       nx = len(xout)
       nz = len(zout)
       LX = np.amax(xout) - np.amin(xout)
       LZ = np.amax(zout) - np.amin(zout)
       
       # Assume that xout and zout are uniformly spaced
       X,Z = np.meshgrid(xout,zout)
       # Initialize the transforms of solutions
       ETA = np.zeros((nx,nz), dtype=complex)
       W = np.zeros((nx,nz), dtype=complex)
       
       # Define the Fourier space for ONE SIDED TRANSFORMS (Smith, 1979)
       kxf = (2*mt.pi/LX) * np.fft.fftfreq(nx) * nx
       kxh = kxf[0:int(nx/2)]
       kx2 = np.power(kxh, 2)
       
       # Define the topography function (Schar Moutnain)
       xsq = np.power(xout, 2)
       cos2x = np.power(np.cos((mt.pi/lc) * xout), 2)
       hx = hc * np.exp(-(1/ac)**2 * xsq) * cos2x
       hk = np.fft.fft(hx)
       
       rho0 = p0/(Rd*T0)
       # Build the transforms of stream function displacement and vertical velocity
       for ii in range(int(nx/2)):
           beta2 = (N/U)**2 - kx2[ii]
           
           if beta2 < 0:
               beta = mt.sqrt(-beta2)
               arge = -beta
           elif beta2 > 0:
               beta = mt.sqrt(beta2)
               arge = 1j * beta
           
           #print(beta2)
           for jj in range(0, nz):
               xid = zout[jj]
               # Compute the smooth, stratified reference fields
               TZ = T0 * mt.exp(N**2/gc * xid)
               EXP = gc**2 / (cp*T0*N**2) * (np.exp(-N**2/gc*xid) - 1.0) + 1.0
               rho = p0/(Rd*TZ) * EXP**(cv/Rd)
               
               # One sided transforms double the energy for the inversion
               thisExp = np.exp(arge * xid)
               thisEta = mt.sqrt(rho0/rho) * hk[ii] * thisExp
               ETA[ii,jj] = thisEta
               W[ii,jj] = 1j * kxf[ii] * U * ETA[ii,jj]
               
       # Recover the solution
       eta = np.real(np.fft.ifft(ETA, axis=0))
       w = 2 * np.real(np.fft.ifft(W, axis=0))
       
       return w.T, ETA, W
       
       
       