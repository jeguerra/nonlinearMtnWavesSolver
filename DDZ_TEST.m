% Test a different way of generating the differentiation matrix for DZ
L = 40000.0;
NZ = 100;
[zlc, ~] = chebdif(NZ, 1);
%
%% Compute using a different implementation

[zo,w]=legslb(NZ);
W = spdiags(w, 0, NZ, NZ);
s = [(0:NZ-2)'+ 0.5;(NZ-1)/2];
S = spdiags(s, 0, NZ, NZ);

[~, HTD] = lepolym(NZ-1, zo);
%[~, HTD] = lepolym(NZ, zo);

%% Compute the coefficients of spectral derivative in matrix form
NM = NZ;
SDIFF = zeros(NM,NM);
SDIFF(NM,NM) = 0.0;
SDIFF(NM-1,NM) = 2 * NM - 1;

k = NM - 1;
for kk = NM-2:-1:1
    A = 2 * k - 3;
    B = 2 * k + 1;
    SDIFF(kk,:) = A / B * SDIFF(kk+2,:);
    SDIFF(kk,kk+1) = A;
    
    k = k - 1;
end

b = 1.0 / L;
DDZ_H1 = HTD' * SDIFF * (S * HTD * W);
figure; surf(DDZ_H1); shading interp;

%% Compute using the built-in algorithm (PRODUCES GARBAGE FOR SIZE > 240)
DDZ_H2 = poldif(zo, 1);
figure; surf(DDZ_H2); shading interp;

%% Test the derivative
Y = (zo.^3) .* (sin(2.0 * pi * zo)).^2;
dY_H1 = DDZ_H1 * Y;
dY_H2 = DDZ_H2 * Y;
figure;
plot(zo, Y, zo, dY_H1, zo, dY_H2);

%% Test spectral differentiation
su1 = ((S * HTD * W) * Y);
dsu1 = SDIFF * su1;
su2 = ledisctran(NZ, zo, w, Y, 0);
dsu2 = lefreqdiff(NZ, su2);
figure;
plot(su1); hold on; plot(su2); legend('Matrix Operator', 'Rercursion');

