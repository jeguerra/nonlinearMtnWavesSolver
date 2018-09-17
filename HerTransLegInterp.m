function [qxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, qxz, NXI, NZI, xint, zint)

%{
Interpolates the 2D field qxz by applying Hermite function transform in the
horizontal and Legendre (barycentric form) interpolation in the vertical.
REFS is a structure with all the necessary data from the original
solution grid. NXI and NZI are the sizes of the new regular grid when xint
and zint are not given in the input.

This method works for fields that were solved using the Hermite Function
derivative in the horizontal and Legendre P. derivative in the vertical.
Fields that DO NOT conform to the Hermite weight function will show errors
when using this method. (i.e. background fields should NOT be interpolated
this way).
%}

%% Interpolate to a regular grid using Hermite and Legendre transforms

% Get the horizontal Hermite nodes
xn = herroots(REFS.NX, 1.0);
b = max(xn) / DS.l2;

% If the input xint grid is a scalar or all zeros make a regular grid with NXI
if (length(xint) == 1 || norm(xint) == 0)
    xint = linspace(min(xn), max(xn), NXI);
    xint = xint';
else
    % Rescale the interpolation grid to the native Hermite node scale
    xint = (xint / max(xint)) * max(xn);
    NXI = length(xint);
end

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
[XINT,ZI] = meshgrid(xint / max(xint) ,zint);

%% Compute the terrain and derivatives
[ht,~] = computeTopoDerivative(REFS.TestCase, xint / b, DS, RAY);

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

%% Get the modal (NX polynomial coefficients) Hermite transform of qxz (NX+1 X NZ)
HT = hefunm(REFS.NX-1, xn);
% Compute the weights and apply to all columns
[~,~,whf] = hegs(REFS.NX);
whfm = repmat(whf, 1, REFS.NZ);
% Apply the weighted transformation
mip = whfm .* qxz';
qmxz = HT * mip;

%% Get the Hermite transform for the interpolated grid and project onto xint
HT = hefunm(REFS.NX-1, xint);
qxzint = HT' * qmxz;

%% Apply Legendre interpolation for the vertical (DMSUITE modification)
[LT,wlf] = poltrans(zn, zint);
wlfm = repmat(wlf, 1, NXI);
qxzint = (LT * (wlfm .* qxzint') ./ (LT * wlfm));

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