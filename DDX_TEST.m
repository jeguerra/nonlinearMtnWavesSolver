% Test a different way of generating the differentiation matrix for DX
L = 800000.0;
NX = 256;
dscale = 0.5 * L;
%
%% Compute using a different implementation
[xo,~] = herdif(NX, 2, dscale, false);
[xh,~] = herdif(NX, 2, dscale, true);

[~,~,w]=hegs(NX);
W = spdiags(w, 0, NX, NX);

[~, HT] = hefunm(NX-1, xo);
[~, HTD] = hefunm(NX, xo);

%% Compute the coefficients of spectral derivative in matrix form
SDIFF = zeros(NX+1,NX);
SDIFF(1,2) = sqrt(0.5);
SDIFF(NX + 1,NX) = -sqrt(NX * 0.5);
SDIFF(NX,NX-1) = -sqrt((NX - 1) * 0.5);

for cc = NX-2:-1:1
    SDIFF(cc+1,cc+2) = sqrt((cc + 1) * 0.5);
    SDIFF(cc+1,cc) = -sqrt(cc * 0.5);
end

b = max(xo) / dscale;
DDX_H1 = b * HTD' * SDIFF * (HT * W);
%rcond(DDX_H)
%rank(DDX_H)
figure; surf(DDX_H1); shading interp;

%% Compute using the built-in algorithm (PRODUCES GARBAGE FOR SIZE > 240)
[xh,DDX_H2] = herdif(256, 1, 0.5*800000.0, true);
figure; surf(DDX_H2); shading interp;