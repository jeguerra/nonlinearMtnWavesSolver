#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:34:06 2020

Convergence data plotted

@author: jeg
"""

import math as mt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Read in the text file
#fname = '/media/jeg/FastDATA/linearMtnWavesSolver/python results/convergence010m_smooth.txt'
fname = '/media/jeg/FastDATA/linearMtnWavesSolver/python results/convergence010m_smooth.txt'
#fname = '/media/jeg/FastDATA/linearMtnWavesSolver/python results/convergence010m_discrete.txt'

con_data = np.loadtxt(fname, delimiter=', ')

# Do an exponential curve fit to the total residual
def func(x, a, b):
       return -b * x + a

lp = 11
xdata = np.arange(0,lp)
ydata = np.log(con_data[0:lp,4])
popt, pcov = curve_fit(func, xdata, ydata, p0=[1.0E-3, 2.0], method='lm')
rate = popt[1]

# Make the nice paper plot
fig = plt.figure(figsize=(12.0, 6.0))
xdata = np.arange(0,con_data.shape[0])
fdata = func(xdata, *popt)

# Make the plots
plt.subplot(1,2,1)
plt.plot(xdata, con_data[:,4], 'kd-')
plt.plot(xdata, np.exp(fdata), 'r--')
plt.yscale('log')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.legend(('Total Residual', 'Convergence Rate = ' + '%.5f' % rate))
plt.xlabel('Newton Iteration')
plt.ylabel('L2-norm of Residual')
plt.title('Total Residual Convergence')

plt.subplot(1,2,2)
plt.plot(xdata, con_data[:,0:4])
plt.yscale('log')
plt.grid(b=None, which='major', axis='both', color='k', linestyle='--', linewidth=0.5)
plt.legend(('u', 'w', 'log-p', 'log-theta'))
plt.xlabel('Newton Iteration')
plt.title('Convergence per Variable')

plt.show()
