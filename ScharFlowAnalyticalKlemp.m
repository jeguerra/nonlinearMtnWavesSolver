function [X,Z,w] = ScharFlowAnalyticalKlemp()

%% Common physical constants
p0 = 1.0E5;
cp = 1004.5;
Rd = 287.0;
cv = cp - Rd;
T0 = 280.0;
g = 9.80616;
U = 10.0;
N = 0.01;
ac = 5000.0;
hc = 250.0;
lc = 4000.0;

%% Schar Mountain Case
%
nx = 256;
nz = 128;
x = [-25000.0 25000.0];
%z = [zsim(1) zsim(end)];
z = [0.0 21000.0];
xspan = linspace(x(1),x(2),nx);
zspan = linspace(z(1),z(2),nz);
%xspan = x(1):100:x(2);
%zspan = z(1):100:z(2);
%}
%xspan = xsim;
%zspan = zsim;
nx = length(xspan);
nz = length(zspan);
LX = abs(xspan(end)-xspan(1));
LZ = abs(zspan(end)-zspan(1));
xd = xspan(1:nx); dx = LX/(nx+1);
zd = zspan(1:nz); dz = LZ/(nz+1);
[X,Z] = meshgrid(xd,zd);
% Initialize the transforms of solutions
ETA = 0.0*X';
W = 0.0*X';

% Define the Fourier space for ONE SIDED TRANSFORMS (Smith, 1979)
kxf = (2*pi/LX)*([0:nx/2-1 -nx/2:-1]);
kxh = (2*pi/LX)*(0:nx-1);
kz = (2*pi/LZ)*(0:nz-1);
kx2 = kxf.^2;

% Define the topography function (Schar Moutnain)
hx = hc * exp(-(1/ac)^2 * xd.^2) .* cos((pi/lc) * xd) .* cos((pi/lc) * xd);
A = 0.25 * ac^2;
KP = 2*pi / lc;
% The analytical transform of the mountain profile (Klemp, 2003) full 2
% sided formula... need only half for the subsequent transform operations
% and is adapted from the reference to include the constant used by Matlab
% in computing fft and ifft (1/nx)
hka = 0.25 * sqrt(pi) * hc * ac * (2 * exp(-A*kxh.^2) + ...
    exp(-A * (KP - kxh).^2) + ...
    exp(-A * (KP + kxh).^2));

%% Check the inverse transform of the mountain profile
%{
HXI = (1.0 / nx) * ifft(hka);
HXI = fftshift(HXI);
figure
plot(xd,real(HXI),xd,hx); grid on; pause;
%}
%% Set the mountain transform
hk = interp1(kxf(1:nx-1),hka(1:nx-1),kxh,'spline');

rho0 = p0/(Rd*T0);
% Build the transforms of stream function displacement and vertical velocity
for ii=1:length(kxh)
    beta2 = N^2/U^2 - kx2(ii);
    if beta2 < 0
        beta = sqrt(-beta2);
        arge = 1i * beta;
    elseif beta2 > 0
        beta = sqrt(beta2);
        arge = beta;
    end
    
    for jj=1:length(kz)
        % Apply stretched coordinate in the transform integrand (Klemp, 2003)
        xid = zspan(end) * (zd(jj) - hx(ii)) / (zspan(end) - hx(ii));
        %xid = zd(jj);
        % Compute the smooth, stratified reference fields
        TZ = T0 * exp(N^2/g * xid);
        EXP = g^2 / (cp*T0*N^2) * (exp(-N^2/g*xid) - 1.0) + 1.0;
        rho = p0/(Rd*TZ) * EXP^(cv/Rd);
        
        % One sided transforms double the energy for the inversion
        ETA(ii,jj) = sqrt(rho0/rho) * hk(ii) * exp(1i * arge * xid);
        %ETA(ii,jj) = 1.0 * hk(ii) * exp(1i * arge * xid);
        W(ii,jj) = 1i * kxf(ii) * U * ETA(ii,jj);
    end
end

eta = real(ifft(ETA,[],1));
w = real(ifft(W,[],1));

% Stretch the grid for output for RAW Tempest results only
for ii=1:length(hx)
    D = zspan(end) - hx(ii);
    st = D / zspan(end);

    coli = Z(1:end-1,ii);
    Z(1:end-1,ii) = st * coli + hx(ii);
end
%
figure
contourf(X,Z,fftshift(eta',2),40); colorbar; grid on;
ylim([0.0 10000.0]);
title('Stream Function Displacement (m) - Schar Mountain Flow');
xlabel('Longitude (m)');
ylabel('Elevation (m)');
figure
contourf(X,Z,fftshift(w',2),40); colorbar; grid on;
ylim([0.0 10000.0]);
title('Vertical Velocity (m/s) - Schar Mountain Flow');
xlabel('Longitude (m)');
ylabel('Elevation (m)');
%}