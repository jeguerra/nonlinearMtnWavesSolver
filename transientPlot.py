#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:25:38 2021

@author: jeg
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage as scm
#import proplot as pplt
from netCDF4 import Dataset
from joblib import Parallel, delayed

plt.rcParams.update({'font.size': 16})

NF = 1600
TIME2STOP = 5.0
m2k = 1.0E-3
runPertb = False
runSGS = False
runPar = False
imgname = '/media/jeguerra/FastDATA/nonlinearMtnWavesSolver/animations/toanimate'
fname = 'Uniform_200m_CS53_SSP104.nc'
m_fid = Dataset(fname, 'r', format="NETCDF4")

times = m_fid.variables['time'][:NF]
timesH = times / 3600.0
X = m2k * m_fid.variables['Xlon'][:,:,0]
Z = m2k * m_fid.variables['Zhgt'][:,:,0]

zbound = 20.0
xbound1 = -20.0
xbound2 = +30.0
zdex = np.nonzero(Z <= zbound)
xdex = np.nonzero((xbound1 <= X) & (X <= xbound2))

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

cmp2plot = 'nipy_spectral'
if runPertb:
       var2plot = th
else:
       var2plot = TH
       
# Get the upper and lower bounds for TH
clim1 = 300.0
clim2 = 380.0 # UNIFORM STRATIFICATION
#clim2 = 510.0 # 3LAYER STRATIFICATION
cline = np.linspace(clim1, clim2, num=40)
print('Plot bounds: ', clim1, clim2)

# Initialize figure
thisFigure = plt.figure(figsize=(18.0, 10.0))
plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.25)

def plotPertb(tt):
       
       thisFigure.gca().clear()
       th2plot = np.ma.getdata(var2plot[tt,:zdex[0].max(),:])
       
       # Median spatial filter
       #th2plot = scm.median_filter(th2plot, size=(4,4))
       
       cflow = plt.contourf(X[:zdex[0].max(),:], 
                            Z[:zdex[0].max(),:], 
                            th2plot, 256, 
                            cmap=cmp2plot, 
                            norm=colors.PowerNorm(gamma=0.4),
                            vmin=clim1, vmax=clim2)
       
       plt.contour(X[:zdex[0].max(),:], 
                   Z[:zdex[0].max(),:], 
                   th2plot, 128,
                   colors='k',
                   norm=colors.PowerNorm(gamma=0.4),
                   vmin=clim1, vmax=clim2)
       
       plt.fill_between(X[0,:], Z[0,:], color='black')
       plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       plt.xlim(xbound1, xbound2)
       plt.ylim(0.0, zbound)
       plt.xlabel('Distance (km)')
       plt.ylabel('Elevation (km)')
       plt.title('Total ' + r'$\theta$ (K)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = times[tt] / 3600.0))
       
       if tt == 0:
              thisFigure.colorbar(cflow, orientation='vertical')
       
       plt.tight_layout()       
       save_file = imgname + f'{tt:04}' + '.jpg'
       thisFigure.savefig(save_file, dpi=200, bbox_inches='tight')
                     
       # Delete stuff
       print('Hour: {thisTime:.2f}'.format(thisTime = timesH[tt]))
       
       return save_file     

#%% Contour animation of perturbation potential temperatures
if runPar:
       print('Attempt parallel processing...')
       imglist = Parallel(n_jobs=8)(delayed(plotPertb)(tt) for tt in range(len(times)))
else:
       print('Run serial processing...')
       imglist = [plotPertb(tt) for tt in range(NF) if timesH[tt] < TIME2STOP]
       