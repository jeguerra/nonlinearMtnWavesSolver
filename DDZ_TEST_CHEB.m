% Test a different way of generating the differentiation matrix for DZ
clc
addpath(genpath('/home/jeguerra/Documents/MATLAB'));
L = 40000.0;
NZ = 400;
%
%% Compute using a different implementation

[zo,w]=cheblb(NZ);
W = spdiags(w, 0, NZ, NZ);

HTD = chebpolym(NZ-1, zo);

%% Compute scaling for the forward transform
s = ones(NZ,1);
for ii=1:NZ-1
    s(ii) = ((HTD(:,ii))' * W * HTD(:,ii))^(-1);
end
s(NZ) = 1.0 / pi;
S = spdiags(s, 0, NZ, NZ);

%% Compute the coefficients of spectral derivative in matrix form
NM = NZ;
SDIFF = zeros(NM,NM);
SDIFF(NM,NM) = 0.0;
SDIFF(NM-1,NM) = 2.0 * NM;

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
end

%% Compute the spectral based derivative
b = 1.0 / L;
DDZ_H1 = HTD' * SDIFF * (S * HTD * W);
%figure; surf(DDZ_H1); shading interp;

%% Compute using the built-in algorithm (PRODUCES GARBAGE FOR SIZE > 240)
DDZ_H2 = chebdiff(NZ-1);
%figure; surf(DDZ_H2); shading interp;

%% Check the difference
%figure; surf(abs(DDZ_H1 - DDZ_H2)); shading interp;

%% Test the derivative on a scaled grid: [-1 1] -> [0 1] (0.5 scaling)
bs = 0.5;
zv = bs * (1.0 - zo);

% Make a test function and its derivative
Y = 4.0 * exp(-2.0 * zv) + cos(4.0 * pi * zv.^2);
DY = -8.0 * exp(-2.0 * zv) - (8.0 * pi * zv) .* sin(4.0 * pi * zv.^2);

% The grid has been rescaled and so must the derivative matrix
dY_H1 = -1.0 / bs * DDZ_H1 * Y;
dY_H2 = -1.0 / bs * DDZ_H2 * Y;

figure;
plot(zv, Y, zv, dY_H1, zv, dY_H2, zv, DY); legend('Function','Spectral Diff','Built-in Diff','Exact');