function [qxzint, XINT, ZINT, ZLINT] = HerTransLagTrans(REFS, DS, qxz, NXI, NZI, xint, zint)

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
bx = max(xn) / DS.l2;

% If the input xint grid is a scalar or all zeros make a regular grid with NXI
if (length(xint) == 1 || norm(xint) == 0.0)
    xint = linspace(min(xn), max(xn), NXI);
else
    % Rescale the interpolation grid to the native Hermite node scale
    xint = (xint / max(xint)) * max(xn);
    NXI = length(xint);
end

% Get the vertical Laguerre nodes
%zn = [0.0; lagroots(REFS.NZ-1)];
zn = lagsrd(REFS.NZ);

% If the input zint grid is a scalar or all zeros make a regular grid with NZI
if (length(zint) == 1 || norm(zint) == 0.0)
    zint = linspace(min(zn), max(zn) , NZI);
else
    % Rescale the input grid to Laguerre node scale
    zint = (zint / max(zint)) * max(zn);
    NZI = length(zint);
end

% Normalize the output grid
[XINT,ZI] = meshgrid(xint / max(xint) ,zint / max(zint));

%% Compute the terrain and derivatives
[ht,~] = computeTopoDerivative(REFS.TestCase, xint / bx, DS);

%% XZ grid for Legendre nodes in the vertical
[HTZL,~] = meshgrid(ht, zint / max(zint));

%% High Order Improved Guellrich coordinate
% 3 parameter function
xi = ZI;
ang = 0.5 * pi * xi;
AR = 1.0E-3;
p = 20;
q = 5;
fxi = exp(-p/q * xi) .* cos(ang).^p + AR * xi .* (1.0 - xi);
dfdxi = -p/q * exp(-p/q * xi) .* cos(ang).^p ...
            -(0.5 * p) * pi * exp(-p/q * xi) .* sin(ang) .* cos(ang).^(p-1) ...
            -AR * (1.0 - 2 * xi);
dzdh = fxi;
dxidz = DS.zH + HTZL .* (dfdxi - fxi);
%sigma = DS.zH * dxidz.^(-1);

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
qxzint = (HT' * qmxz)';

%% Get the modal (NZ polynomial coefficients) Laguerre transform of qxz (NX+1 X NZ)
LT = lafunm(REFS.NZ-1, zn);
% Compute the weights and apply to all columns
[~,~,wlf] = lagsrd(REFS.NZ);
wlfm = repmat(wlf, 1, length(xint));
% Apply the weighted transformation
mip = wlfm .* qxzint;
qmxz = LT * mip;

%% Get the Laguerre transform for the interpolated grid and project onto zint
LT = lafunm(REFS.NZ-1, zint);
qxzint = LT' * qmxz;

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