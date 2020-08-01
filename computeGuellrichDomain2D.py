#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:23:58 2019

@author: TempestGuerra
"""

import numpy as np
import math as mt
#import matplotlib.pyplot as plt

def computeTerrainDecayFunctions(xi, ang, StaticSolve):
       
       if StaticSolve:
              # Nominal Hybrid coordinate
              m = 0.2
              mi = 1.0 / m
              dzdh = np.sinh(mi * (1.0 - xi)) / np.sinh(mi)
              d_dzdh_dxi = -mi * np.cosh(mi * (1.0 - xi)) / np.sinh(mi)
       else:
              # First pass [A=0.3, p=20, m=0.2]
              # Second pass [A=0.4, p=10, m=0.25]
              # Third pass [A=0.25, p=25, m=0.25]
              A = 0.25
              p = 25
              
              # Guellrich improvement to hybrid coordinate
              cosvar = np.power(np.cos(A * ang), p)
              tanvard = A * mt.pi * np.tan(A * ang)
              
              # Guellrich Hybrid coordinate
              m = 0.25
              mi = 1.0 / m
              hybrid = np.sinh(mi * (1.0 - xi)) / np.sinh(mi)
              dhybrid = -mi * np.cosh(mi * (1.0 - xi)) / np.sinh(mi)
              dzdh = cosvar * np.sinh(mi * (1.0 - xi)) / np.sinh(mi)
              d_dzdh_dxi = cosvar * (dhybrid - p * hybrid * tanvard) 
       
       return dzdh, d_dzdh_dxi

def computeGuellrichDomain2D(DIMS, REFS, zRay, hx, dhdx, StaticSolve):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       
       # input REFS = [x, z, HFM, whf, CPM, wcp]
       x = REFS[0]
       z = REFS[1]
       
       # Compute the flat XZ mesh (computational domain)
       HTZL, dummy = np.meshgrid(hx,z)
       XL, ZL = np.meshgrid(x,z)
       
       # High Order Improved Guellrich coordinate 3 parameter function
       xi = 1.0 / ZH * ZL
       ang = 1.0/3.0 * mt.pi * xi
       dzdh, d_dzdh_dxi = computeTerrainDecayFunctions(xi, ang)
       
       dxidz = ZH + (HTZL * d_dzdh_dxi)
       sigma = ZH * np.reciprocal(dxidz)
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       DZT = np.zeros((NZ,NX))
       
       for rr in range(NZ):
              ZTL[rr,:] = (dzdh[rr,0] * hx) + ZL[rr,:]
              DZT[rr,:] = dzdh[rr,0] * dhdx
              
       #plt.plot(z, dzdh[:,0])
       
       # Compute the coordinate surface at edge of Rayleigh layer
       xi = 1.0 / ZH * zRay
       ang = mt.pi * zRay
       dzdh, d_dzdh_dxi = computeTerrainDecayFunctions(xi, ang)
       
       ZRL = (dzdh * hx) + zRay
       
       # Compute the local grid lengths at each node
       DXM = np.zeros((NZ,NX))
       DZM = np.zeros((NZ,NX))
       
       for ii in range(NZ):
              xdiff = np.diff(XL[ii,:])
              DXM[ii,:] = np.concatenate((np.expand_dims(xdiff[0],0), xdiff)) 
       
       for jj in range(NX):
              zdiff = np.diff(ZTL[:,jj])
              DZM[:,jj] = np.concatenate((np.expand_dims(zdiff[0],0), zdiff)) 
       
       return XL, ZTL, DZT, sigma, ZRL, DXM, DZM

def computeStretchedDomain2D(DIMS, REFS, zRay, hx, dhdx):
       # Get data from DIMS and REFS
       ZH = DIMS[2]
       NX = DIMS[3] + 1
       NZ = DIMS[4]
       
       # Get REFS data
       x = REFS[0]
       z = REFS[1]
       DDX_1D = REFS[2]
       
       # Compute the flat XZ mesh
       DZT, dummy = np.meshgrid(dhdx,z);
       XL, ZL = np.meshgrid(x,z);
       
       # Make the global array of terrain height and slope features
       ZTL = np.zeros((NZ,NX))
       
       sigma = np.ones((NZ,NX))
       for cc in range(NX):
              thisZH = ZH - hx[cc]
              sigma[:,cc] *= (ZH / thisZH)
              ZTL[:,cc] = ZL[:,cc] * thisZH / ZH
              ZTL[:,cc] += hx[cc]
       
       # Compute the terrain derivatives       
       for rr in range(1,NZ):
              DZT[rr,:] = DDX_1D.dot(ZTL[rr,:] - z[rr])
              
       # Compute the coordinate surface at the edge of the Rayleigh layer
       ZRL = (1.0 - zRay / ZH) * hx + zRay
       
       return XL, ZTL, DZT, sigma, ZRL