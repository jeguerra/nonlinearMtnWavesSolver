% Test a different way of generating the differentiation matrix for DZ
clc
addpath(genpath('/home/jeguerra/Documents/MATLAB'));
L = 40000.0;
NZ = 100;
%
%% Compute using a different implementation

[zo,w]=cheblb(NZ);
W = spdiags(w, 0, NZ, NZ);

HTD = chebpolym(NZ-1, zo);

%% Compute the coefficients of spectral derivative in matrix form
NM = NZ;
SDIFF = zeros(NM,NM);
SDIFF(NM,NM) = 0.0;
SDIFF(NM-1,NM) = 2 * NM;

k = NM - 1;
for kk = NM-2:-1:1
    A = 2 * kk;
    B = 1;
    if kk > 1
      c = 1.0;
    else
      c = 2.0;
    end
    SDIFF(kk,:) = B / c * SDIFF(kk+2,:);
    SDIFF(kk,kk+1) = A / c;
    
    k = k - 1;
end

b = 1.0 / L;
DDZ_H1 = HTD' * SDIFF * (HTD * W);
figure; surf(DDZ_H1); shading interp;

%% Compute using the built-in algorithm (PRODUCES GARBAGE FOR SIZE > 240)
DDZ_H2 = chebdiff(NZ-1);
figure; surf(DDZ_H2); shading interp;

%% Test the derivative
Y = (4.0 * zo) .* cos(2.0 * pi * zo);
dY_H1 = DDZ_H1 * Y;
dY_H2 = DDZ_H2 * Y;
figure;
plot(zo, Y, zo, dY_H1, zo, dY_H2); legend('Function','Spectral Diff','Built-in Diff');