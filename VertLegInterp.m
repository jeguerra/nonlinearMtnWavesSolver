function [qxzint, XINT, ZINT, ZLINT] = VertLegInterp(REFS, DS, RAY, qxz, NXI, NZI, zint)

%{
Interpolates the 2D field qxz by applying Legendre (barycentric form) interpolation in the vertical.
REFS is a structure with all the necessary data from the original
solution grid. NXI and NZI are the sizes of the new regular grid when xint
and zint are not given in the input.

This method works for fields that were solved using the Hermite Function
derivative in the horizontal and Legendre P. derivative in the vertical.
Fields that DO NOT conform to the Hermite weight function will show errors
when using this method. (i.e. background fields should NOT be interpolated
this way).
%}

% If the input xint grid is a scalar or all zeros make a regular grid with NXI
xint = linspace(DS.l1, DS.l2, NXI);
xint = xint';

% Get the vertical Chebyshev nodes
[zn,~] = chebdif(REFS.NZ, 1);
zn = 0.5 * (zn + 1.0);

% If the input zint grid is a scalar or all zeros make a regular grid with NZI
if (length(zint) == 1 || norm(zint) == 0)
    zint = linspace(0.0, 1.0 ,NZI);
    zint = zint';
else
    % Rescale the input grid to [0 1]
    zint = zint / max(zint);
    NZI = length(zint);
end

% Normalize the output grid
[XINT,ZI] = meshgrid(xint ,zint);

%% Compute the terrain and derivatives
[ht,~] = computeTopoDerivative(REFS.TestCase, xint', DS, RAY);

%% XZ grid for Legendre nodes in the vertical
[HTZL,~] = meshgrid(ht, DS.zH * zint);

%% High Order Improved Guellrich coordinate
% 3 parameter function
xi = ZI;
ang = 0.5 * pi * xi;
AR = 1.0E-3;
p = 20;
q = 5;
fxi = exp(-p/q * xi) .* cos(ang).^p + AR * xi .* (1.0 - xi);
dzdh = fxi;

% Adjust Z with terrain following coords
ZINT = (dzdh .* HTZL) + ZI * DS.zH;
ZLINT = ZI * DS.zH;

%% Apply Legendre interpolation for the vertical (DMSUITE modification)
[LT,wlf] = poltrans(zn, zint);
wlfm = repmat(wlf, 1, NXI);
%size(LT)
%size(wlfm)
%size(qxz)
qxzint = (LT * (wlfm .* qxz) ./ (LT * wlfm));

%}
%{
%% DEBUGGING PLOTS
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
%contourf(XI, ZTI, qxzint,31); colorbar;
surf(XI, ZTI, qxzint); colorbar; shading faceted;
%hold on;
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
%contourf(XI, ZTI, qxzint,31); colorbar;
surf(REFS.XL, REFS.ZTL, qxz); colorbar; shading faceted;
%}