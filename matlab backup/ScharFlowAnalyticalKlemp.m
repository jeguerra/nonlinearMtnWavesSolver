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
nx = 1024;
nz = 256;
x = [-50000.0 50000.0];
z = [0.0 25000.0];
xd = linspace(x(1),x(2),nx);
zd = linspace(z(1),z(2),nz);
nx = length(xd);
nz = length(zd);
LX = abs(x(2)-x(1));
LZ = abs(z(2)-zd(1));
[X,Z] = meshgrid(xd,zd);
% Initialize the transforms of solutions
ETA = 0.0*X';
W = 0.0*X';

% Define the Fourier space for ONE SIDED TRANSFORMS (Smith, 1979)
kxf = (2*pi/LX)*([0:nx/2-1 -nx/2:-1]);
kxh = (2*pi/LX)*(0:nx/2-1);
kx2 = kxh.^2;

% Define the topography function (Schar Moutnain)
hx = hc * exp(-(1/ac)^2 * xd.^2) .* cos((pi/lc) * xd) .* cos((pi/lc) * xd);
hk = fft(hx);

%% Check the inverse transform of the mountain profile
%{
% The analytical transform of the mountain profile (Klemp, 2003) full 2
% sided formula... need only half for the subsequent transform operations
% and is adapted from the reference to include the constant used by Matlab
% in computing fft and ifft (1/nx)
A = 0.25 * ac^2;
KP = 2*pi / lc;
hka = 0.25 * hc * ac / nx * (2 * exp(-A*kxf.^2) + ...
    exp(-A * (KP - kxf).^2) + ...
    exp(-A * (KP + kxf).^2));
HXF = ifft(hkf);
HXA = fftshift(ifft(hka));
figure
plot(xd,real(HXF),xd,real(HXA)); grid on; pause;
%}

rho0 = p0/(Rd*T0);
% Build the transforms of stream function displacement and vertical velocity
for ii=1:nx/2
    beta2 = N^2/U^2 - kx2(ii);
    if beta2 < 0
        beta = sqrt(-beta2);
        arge = -beta;
    elseif beta2 > 0
        beta = sqrt(beta2);
        arge = 1i * beta;
    end
    
    for jj=1:nz
        xid = zd(jj);
        % Compute the smooth, stratified reference fields
        TZ = T0 * exp(N^2/g * xid);
        EXP = g^2 / (cp*T0*N^2) * (exp(-N^2/g*xid) - 1.0) + 1.0;
        rho = p0/(Rd*TZ) * EXP^(cv/Rd);
        
        % One sided transforms double the energy for the inversion
        ETA(ii,jj) = sqrt(rho0/rho) * hk(ii) * exp(arge * xid);
        W(ii,jj) = 1i * kxf(ii) * U * ETA(ii,jj);
    end
end

eta = real(ifft(ETA,[],1));
w = 2 * real(ifft(W,[],1));
%%
figure
contourf(X,Z,eta',40); colorbar; grid on;
ylim([0.0 20000.0]);
title('Stream Function Displacement (m) - Schar Mountain Flow');
xlabel('Longitude (m)');
ylabel('Elevation (m)');
figure
contourf(X,Z,w',40); colorbar; grid on;
ylim([0.0 20000.0]);
title('Vertical Velocity (m/s) - Schar Mountain Flow');
xlabel('Longitude (m)');
ylabel('Elevation (m)');
figure
surf(X,Z,w'); colorbar; grid on;
ylim([0.0 5000.0]);
title('Vertical Velocity (m/s) - Schar Mountain Flow');
xlabel('Longitude (m)');
ylabel('Elevation (m)');
%}
%% Save the data
%
close all;
fileStore = 'AnalyticalSchar.mat';
save(fileStore);
%}