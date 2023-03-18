#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:25:38 2021

@author: jeg
"""
import os
import imageio.v2 as imageio
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from joblib import Parallel, delayed

m2k = 1.0E-3
fname = 'Simulation2View1.nc'
m_fid = Dataset(fname, 'r', format="NETCDF4")

times = m_fid.variables['time'][:]
X = m_fid.variables['Xlon'][:,:,0]
Z = m_fid.variables['Zhgt'][:,:,0]

U = m_fid.variables['U'][:,:,0]
LNP = m_fid.variables['LNP'][:,:,0]
LNT = m_fid.variables['LNT'][:,:,0]

u = m_fid.variables['u'][:,:,:,0]
w = m_fid.variables['w'][:,:,:,0]
lnp = m_fid.variables['ln_p'][:,:,:,0]
lnt = m_fid.variables['ln_t'][:,:,:,0]

dlnt = m_fid.variables['Dln_tDt'][:,:,:,0]

# Compute the total and perturbation potential temperature
TH = np.exp(LNT + lnt)
th = TH - np.exp(LNT)

# Get the upper and lower bounds for TH
clim1 = th.min()
clim2 = th.max()

imgname = 'toanimate.png'
THname = 'TotalPT.gif'
thname = 'PerturbationPT.gif'
sgsname = 'SGS-PT.gif'

runPertb = True
runSGS = False

runPar = False
runSer = True

fig = plt.figure(figsize=(16.0, 8.0))

def plotPertb(tt):
       plt.gca().clear()
       plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.25)
       plt.contourf(1.0E-3*X, 1.0E-3*Z, th[tt,:,:], 101, cmap=cm.prism, vmin=clim1, vmax=clim2)
       
       #norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
       #plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic), format='%.2e', cax=plt.gca())
       
       plt.contour(m2k * X, m2k * Z, TH[tt,:,:], 21, colors='white', linewidths=1.0)
       
       plt.fill_between(m2k * X[0,:], m2k * Z[0,:], color='white')
       plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       plt.xlim(-50.0, 50.0)
       plt.ylim(0.0, 25.0)
       plt.title('Total ' + r'$\theta$ and $\Delta \theta$ (K)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = times[tt] / 3600.0))
       plt.tight_layout()
       # Save out the image
       plt.savefig(imgname)
       
       # Get the current image
       image = imageio.imread(imgname)
                     
       # Delete stuff
       print('Hour: {timeH:.2f}'.format(timeH = times[tt] / 3600.0))
       
       return image

#%% Contour animation of perturbation potential temperatures
if runPertb:
       # Serial processing
       if runSer:
              print('Run serial processing...')
              imglist = []
              for tt in range(len(times)):
                     image = plotPertb(tt)
                     imglist.append(image)
              
              
       # Parallel processing
       if runPar:
              print('Attempt parallel processing...')
              imglist = Parallel(n_jobs=4)(delayed(plotPertb)(tt) for tt in range(len(times)))
       
       imageio.mimsave(thname, imglist, fps=20)
       
#%% Contour animation of the normalized SGS
if runSGS:
       imglist = []
       for tt in range(len(times)):
              
              if tt == 0:
                     dt = times[tt+1] - times[tt]
              else:
                     dt = times[tt] - times[tt-1]
                     
              if dt < 1.0E-10:
                     continue
              
              # Compute the SGS
              if tt == 0:
                     q = np.zeros(th[tt,:,:].shape)
                     norm = 1.0
              elif tt == 1:
                     q = 1.0 / dt * (th[tt,:,:] - th[tt-1,:,:]) - \
                            0.5 * (dlnt[tt-1,:,:] + dlnt[tt+1,:,:])
                     norm = 1.0 / np.amax(np.abs(q))
              elif tt == len(times)-1:
                     q = 1.0 / dt * (th[tt,:,:] - th[tt-1,:,:]) - dlnt[tt,:,:]
                     norm = 1.0 / np.amax(np.abs(q))
              else:
                     q = 0.5 / dt * (3.0 * th[tt,:,:] - 4.0 * th[tt-1,:,:] + th[tt-2,:,:]) - \
                            0.5 * (dlnt[tt-1,:,:] + dlnt[tt+1,:,:])
                     norm = 1.0 / np.amax(np.abs(q))
                     
              qSGS = norm * q
              
              fig = plt.figure(figsize=(16.0, 8.0))
              plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.25)
              
              cc = plt.contourf(1.0E-3*X, 1.0E-3*Z, qSGS[:,:], 101, cmap=cm.seismic, vmin=-1.0, vmax=1.0)
              
              norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
              plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.seismic), format='%.2e')
              
              plt.fill_between(m2k * X[0,:], m2k * Z[0,:], color='black')
              plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
              plt.xlim(-50.0, 50.0)
              plt.ylim(0.0, 25.0)
              plt.title('Normalized SGS: ' + r'$\theta$' + \
                        ' Hour: {timeH:.2f}'.format(timeH = times[tt] / 3600.0))
              plt.tight_layout()
              # Save out the image
              plt.savefig(imgname)
              
              # Get the current image and add to gif list
              image = imageio.imread(imgname)
              imglist.append(image)
                            
              # Delete stuff
              os.remove(imgname)
              plt.close('all')
              del(fig)
       
       imageio.mimsave(sgsname, imglist, fps=10)