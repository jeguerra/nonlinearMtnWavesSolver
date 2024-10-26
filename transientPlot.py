#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:25:38 2021

@author: jeg
"""
from PIL import ImageFile
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
#import proplot as pplt
from netCDF4 import Dataset
from joblib import Parallel, delayed

plt.rcParams.update({'font.size': 16})
ImageFile.LOAD_TRUNCATED_IMAGES = True

TIME2STOP = 5.0
m2k = 1.0E-3
runPertb = False
runSGS = False
runPar = False
#test_type = 'Uniform'
test_type = '3Layer'
imgname = '/media/jeguerra/FastDATA/nonlinearMtnWavesSolver/animations/toanimate'
fname = test_type + '_170m_CS35_1Em15_RES.nc'
m_fid = Dataset(fname, 'r', format="NETCDF4")

zbound = 20.0
xbound1 = -20.0
xbound2 = +30.0

times = m_fid.variables['time'][:]
timesH = times / 3600.0
tdex = np.nonzero(timesH <= TIME2STOP)[0]

XT = m2k * m_fid.variables['Xlon'][:,:,0]
ZT = m2k * m_fid.variables['Zhgt'][:,:,0]
zdex = np.nonzero(ZT <= zbound)[0]
xdex = np.nonzero((XT >= xbound1) & (XT <= xbound2))[1]

X = XT[:zdex.max(),xdex[0]:xdex[-1]]
Z = ZT[:zdex.max(),xdex[0]:xdex[-1]]

u = m_fid.variables['u'][tdex,:zdex.max(),xdex[0]:xdex[-1],0]
w = m_fid.variables['w'][tdex,:zdex.max(),xdex[0]:xdex[-1],0]
lnt = m_fid.variables['ln_t'][tdex,:zdex.max(),xdex[0]:xdex[-1],0]
lnt_sgs = m_fid.variables['SGSln_t'][tdex,:zdex.max(),xdex[0]:xdex[-1],0]

# Compute the total and perturbation potential temperature
TH = np.exp(lnt)

# Set up plotting bounds
if test_type == 'Uniform':
       tlim1 = 299.0
       tlim2 = 375.0
       slim1 = -5.0E-4
       slim2 = +5.0E-4
       dlim1 = -25.0
       dlim2 = +25.0
       vlim1 = -160.0
       vlim2 = +160.0
else:
       tlim1 = 299.0
       tlim2 = 510.0
       slim1 = -5.0E-4
       slim2 = +5.0E-4
       dlim1 = -25.0
       dlim2 = +25.0
       vlim1 = -160.0
       vlim2 = +160.0

# Get the post-processing derivative operators
import computeDerivativeMatrix as derv
DDXP = sps.csr_array(derv.computeCompactFiniteDiffDerivativeMatrix1(XT[-1,xdex[0]:xdex[-1]], 4))
DDZP = sps.csr_array(derv.computeCompactFiniteDiffDerivativeMatrix1(ZT[:zdex.max(),0], 4))
DZDX = (DDXP @ Z.T).T

# Initialize figure
thisFigure, theseAxes = plt.subplots(nrows=2,ncols=2,
                                     sharex='col',sharey='row',
                                     figsize=(24.0, 12.0))
plt.grid(visible=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.25)

def plotPertb(tt):
       
       # Main Theta plot
       theseAxes[0,0].clear()
       th2plot = np.ma.getdata(TH[tt,:,:])
       cflow1 = theseAxes[0,0].contourf(X, Z, 
                            th2plot, 128, 
                            cmap='nipy_spectral', 
                            norm=colors.PowerNorm(gamma=0.4),
                            vmin=tlim1, vmax=tlim2)
       
       theseAxes[0,0].contour(X, Z, 
                   th2plot, 80,
                   colors='k',
                   norm=colors.PowerNorm(gamma=0.4),
                   vmin=tlim1, vmax=tlim2)
       
       theseAxes[0,0].fill_between(X[0,:], Z[0,:], color='black')
       theseAxes[0,0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       theseAxes[0,0].set_ylabel('Elevation (km)')
       theseAxes[0,0].set_title('Total ' + r'$\theta$ (K)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = timesH[tt]))
              
       # Theta DynSGS plot
       theseAxes[0,1].clear()
       sg2plot = np.ma.getdata(lnt_sgs[tt,:,:])
       cflow2 = theseAxes[0,1].contourf(X, Z, 
                            sg2plot, 128, 
                            cmap='seismic',
                            vmin=slim1, vmax=slim2)
       
       theseAxes[0,1].fill_between(X[0,:], Z[0,:], color='black')
       theseAxes[0,1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       theseAxes[0,1].set_title('SGS Tendency ' + r'$\frac{\partial \theta}{\partial t}$ (K/t)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = timesH[tt]))
              
       # Divergence plot
       up = np.ma.getdata(u[tt,:,:])
       PuPz = DDZP @ up
       PuPx = (DDXP @ up.T).T - DZDX * PuPz
       wp = np.ma.getdata(w[tt,:,:])
       PwPz = DDZP @ wp
       DivUW = (PuPx + PwPz)
       theseAxes[1,0].clear()
       cflow3 = theseAxes[1,0].contourf(X, Z, 
                            DivUW, 128, 
                            cmap='seismic',
                            vmin=dlim1, vmax=dlim2)
       
       theseAxes[1,0].fill_between(X[0,:], Z[0,:], color='black')
       theseAxes[1,0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       theseAxes[1,0].set_xlabel('Distance (km)')
       theseAxes[1,0].set_ylabel('Elevation (km)')
       theseAxes[1,0].set_title('Divergence Field (1/t)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = timesH[tt]))
              
       # Vorticity plot
       PwPz = DDZP @ wp
       PwPx = (DDXP @ wp.T).T - DZDX * PwPz
       VrtUW = (PuPz - PwPx)
       theseAxes[1,1].clear()
       theseAxes[1,1].contourf(X, Z, 
                            VrtUW, 128,
                            cmap='seismic',
                            vmin=vlim1, vmax=vlim2)
       
       theseAxes[1,1].fill_between(X[0,:], Z[0,:], color='black')
       theseAxes[1,1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
       theseAxes[1,1].set_xlabel('Distance (km)')
       theseAxes[1,1].set_title('Vorticity Field (1/t)' + \
                 ' Hour: {timeH:.2f}'.format(timeH = timesH[tt]))
       
       if tt == 0:
              print('Initial data ranges:')
              print(th2plot.min().min(), th2plot.max().max())
              print(sg2plot.min().min(), sg2plot.max().max())
              print(DivUW.min().min(), DivUW.max().max())
              print(VrtUW.min().min(), VrtUW.max().max())
              
              norm = colors.Normalize(vmin=tlim1, vmax=tlim2)
              cflow4 = cm.ScalarMappable(norm=norm, cmap='nipy_spectral')
              thisFigure.colorbar(cflow1, orientation='vertical', 
                                  ax=theseAxes[0,0])
              
              norm = colors.Normalize(vmin=slim1, vmax=slim2)
              cflow2 = cm.ScalarMappable(norm=norm, cmap='seismic')
              thisFigure.colorbar(cflow2, orientation='vertical',
                                  ax=theseAxes[0,1])
              
              norm = colors.Normalize(vmin=dlim1, vmax=dlim2)
              cflow3 = cm.ScalarMappable(norm=norm, cmap='seismic')
              thisFigure.colorbar(cflow3, orientation='vertical',
                                  ax=theseAxes[1,0])
              
              norm = colors.Normalize(vmin=vlim1, vmax=vlim2)
              cflow4 = cm.ScalarMappable(norm=norm, cmap='seismic')
              thisFigure.colorbar(cflow4, orientation='vertical',
                                  ax=theseAxes[1,1])
       elif tt == tdex[-1]:
              print('Final data ranges:')
              print(th2plot.min().min(), th2plot.max().max())
              print(sg2plot.min().min(), sg2plot.max().max())
              print(DivUW.min().min(), DivUW.max().max())
              print(VrtUW.min().min(), VrtUW.max().max())
       
       thisFigure.tight_layout()       
       save_file = imgname + f'{tt:04}' + '.jpg'
       thisFigure.savefig(save_file, dpi=180, bbox_inches='tight')
                     
       # Delete stuff
       print('Hour: {thisTime:.2f}'.format(thisTime = timesH[tt]))
       
       return save_file     

#%% Contour animation of perturbation potential temperatures
if runPar:
       print('Attempt parallel processing...')
       imglist = Parallel(n_jobs=8)(delayed(plotPertb)(tt) for tt in range(len(times)))
else:
       print('Run serial processing...')
       imglist = [plotPertb(tt) for tt in tdex if timesH[tt] < TIME2STOP]
       