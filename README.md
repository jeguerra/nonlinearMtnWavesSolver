# linearMtnWavesSolver
MATLAB tools to solve mountain wave flows

Main run scripts:

# Runs the LnP-LnTheta (u, w, ln(p), ln(th)) model in steady state
Euler2DTerrainFollowingSpectralEllipticCBC.m

# Runs the conservative (ru, rw, r, rt) model in steady state
Euler2DTerrainFollowingSpectralEllipticCBC_FluxForm.m

# Runs the LnP-LnTheta model in steady state (FFT in X direction)
Euler2DTerrainFollowingSpectralEllipticFFT.m

# Runs the LnP-LnTheta model transient using the SSP RK53 explicit method
Euler2DTerrainFollowingSpectralSSPRK53CBC.m

INSTALL THE ExternalUtilities.zip PACKAGES IN THE MATLAB PATH. 

REFERENCES:

Shen, J., T. Tang, and L-L. Wang (2011). Spectral Methods: Algorithms, Analysis, and Applications. 1961 Springer-Verlag Berlin Heidelberg. DOI: 10.1007/978-3-540-71041-7

Weideman, J. A. and S. C. Reddy (2000). “AMATLAB DifferentiationMatrix Suite”. In: ACMTrans. 2012 Math. Softw. 26.4, pp. 465–519. ISSN: 0098-3500. DOI: 10.1145/365723.365727




