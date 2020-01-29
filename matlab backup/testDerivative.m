clc
clear
close all

NO = 80;
NX = 200;

L = 2 * pi;
[xh,~] = herdif(NX, 2, L, false);
[xphys,ddx_phys] = herdif(NX, 2, L, true);

xfft = linspace(-L,L,NX);

[~,~,w]=hegs(NX);
W = spdiags(w, 0, NX, NX);

[~, HT] = hefunm(NO-1, xh);
[~, HTD] = hefunm(NO, xh);

%% SIMPLE FUNCTION TO DIFFERENTIATE EXACTLY WITH HERMITE FUNCTIONS
fun = exp(-(xphys).^2) .* sin(1.5 * xphys);
funFFT = exp(-(xfft).^2) .* sin(1.5 * xfft);

%% Compute the coefficients of spectral derivative in matrix form
SDIFF = zeros(NO+1,NO);
SDIFF(1,2) = sqrt(0.5);
SDIFF(NO + 1,NO) = -sqrt(NO * 0.5);
SDIFF(NO,NO-1) = -sqrt((NO - 1) * 0.5);

for cc = NO-2:-1:1
    SDIFF(cc+1,cc+2) = sqrt((cc + 1) * 0.5);
    SDIFF(cc+1,cc) = -sqrt(cc * 0.5);
end

%% Compute derivatives 3 different ways...
% DERIVATIVE IN PHYSICAL SPACE
dfun_phys = ddx_phys(:,:,1) * fun;
% DERIVATIVE IN FREQ SPACE (MATRIX FORM) with domain scaling!
b = max(xh) / L;
dfun_moda = b * HTD' * SDIFF * (HT * (W * fun));
% DERIVATIVE IN FREQ SPACE (DEFAULT FORM)
funh = hefdisctran(NO,xh,w,fun,0);
dfunh = heffreqdiff(NO, funh);
dfun_freq = b * hefdisctran(NO+1,xh,w,dfunh,1);
% DERIVATIVE WITH FFT
dfun_fft = b / L * fourdifft(funFFT,1);

%% THE DERIVATIVE NEEDS TO BE PROPERLY SCALED ACCORDING TO THE DOMAIN!
plot(xphys, fun, ...
     xphys, dfun_phys, ...
     xphys, dfun_moda, 'o-', ...
     xphys, dfun_freq, 's-', ...
     xfft, dfun_fft, '*-'); grid on;
legend('Function','Spatial Derivative','Spectral Derivative Matrix','Spectral Derivative Transforms');