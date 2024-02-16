#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:25:38 2021

@author: jeg
"""
import os
import time
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import bottleneck as bn
import scipy.linalg as scl
import scipy.ndimage as scm
import matplotlib as mpl
from matplotlib import cm
import seaborn as sns
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from joblib import Parallel, delayed

plt.rcParams.update({'font.size': 16})

NF = 1080
m2k = 1.0E-3
runPertb = False
runSGS = False
runPar = False
imgname = '/media/jeguerra/FastDATA/linearMtnWavesSolver/animations/toanimate'
fname = 'Simulation2View_mkernel.nc'
m_fid = Dataset(fname, 'r', format="NETCDF4")

times = m_fid.variables['time'][:NF]
X = m2k * m_fid.variables['Xlon'][:,:,0]
Z = m2k * m_fid.variables['Zhgt'][:,:,0]

zdex = np.nonzero(Z <= 21.0)
xdex = np.nonzero((-30.0 <= X) & (X <= +50.0))

#U = m_fid.variables['U'][:,:,0]
#LNP = m_fid.variables['LNP'][:,:,0]
LNT = m_fid.variables['LNT'][:,:,0]

#u = m_fid.variables['u'][:,:,:,0]
#w = m_fid.variables['w'][:,:,:,0]
#lnp = m_fid.variables['ln_p'][:,:,:,0]
lnt = m_fid.variables['ln_t'][:NF,:,:,0]

# Compute the total and perturbation potential temperature
TH = np.exp(lnt)
th = TH - np.exp(LNT)

if runPertb:
       var2plot = th
       cmp2plot = 'nipy_spectral'
       out_name = 'PerturbationPT01.gif'
else:
       var2plot = TH
       cmp2plot = sns.color_palette('Spectral_r', as_cmap=True)
       out_name = 'TotalPT01.gif'

# Get the upper and lower bounds for TH
clim1 = 300.0
clim2 = 450.0
print('Plot bounds: ', clim1, clim2)

# Initialize figure
thisFigure = plt.figure(figsize=(18.0, 10.0))
plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.25)

def plotPertb(tt):
       
       thisFigure.gca().clear()
       th2plot = np.ma.getdata(var2plot[tt,:zdex[0].max(),:])
       '''
       # Time filtering (Useless for stationary modes)
       if tt > 1 and tt < times.shape[0] - 1:
              th2plot = np.ma.getdata(var2plot[tt-1,:,:]) + \
                        np.ma.getdata(var2plot[tt,:,:]) + \
                        np.ma.getdata(var2plot[tt+1,:,:])
              th2plot *= 1.0 / 3.0
       '''
       # SVD spatial filter
       '''
       svdq = scl.svd(th2plot, full_matrices=False)
       quantile_filter = np.nanquantile(svdq[1], 0.75)
       print(svdq[1].max(), svdq[1].min(), quantile_filter)
       sdex = np.nonzero(svdq[1] >= quantile_filter)
       qf = svdq[0][:,sdex].dot(np.diag(svdq[1][sdex]))
       th2plot = qf.dot(svdq[2][sdex,:])[:,0,0,:]
       '''
       # Median spatial filter
       th2plot = scm.median_filter(th2plot, size=(8,4))
       
       cflow = plt.contourf(X[:zdex[0].max(),:], 
                            Z[:zdex[0].max(),:], 
                            th2plot, 512, cmap=cmp2plot, vmin=clim1, vmax=clim2)
       plt.contour(X[:zdex[0].max(),:], 
                   Z[:zdex[0].max(),:], 
                   th2plot, 36, colors='k', vmin=clim1, vmax=clim2)
       
       plt.fill_between(X[0,:], Z[0,:], color='black')
       plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       plt.xlim(-25.0, 50.0)
       plt.ylim(0.0, 20.0)
       plt.xlabel('Distance (km)')
       plt.ylabel('Elevation (km)')
       plt.title('Total ' + r'$\theta$ (K)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = times[tt] / 3600.0))
       
       if tt == 0:
              thisFigure.colorbar(cflow, location='bottom')
       
       plt.tight_layout()
       #plt.show()
       
       save_file = imgname + f'{tt:04}' + '.jpg'
       #input('Image Check')
       
       # Save out the image
       thisFigure.savefig(save_file)
       #plt.close(fig=thisFigure)
       
       # Get the current image
       #image = Image.open(save_file)
                     
       # Delete stuff
       print('Hour: {timeH:.2f}'.format(timeH = times[tt] / 3600.0))
       
       return save_file
       
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
              
              cc = plt.contourf(1.0E-3*X, 1.0E-3*Z, qSGS[:,:], 64, cmap=cm.seismic, vmin=-1.0, vmax=1.0)
              
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
              time.sleep(0.01)
              
              # Get the current image and add to gif list
              image = Image.open(imgname)
              imglist.append(image)
                            
              # Delete stuff
              os.remove(imgname)
              plt.close('all')
              del(fig)
#%% Contour animation of perturbation potential temperatures
else:    
       # Parallel processing
       if runPar:
              print('Attempt parallel processing...')
              imglist = Parallel(n_jobs=8)(delayed(plotPertb)(tt) for tt in range(len(times)))
       else:
              print('Run serial processing...')
              #imglist = [plotPertb(tt) for tt in range(len(times))]
              imglist = [plotPertb(tt) for tt in range(NF)]
       
#imglist[0].save(out_name,append_images=imglist[1:], save_all=True, optimize=True, duration=30, loop=0)