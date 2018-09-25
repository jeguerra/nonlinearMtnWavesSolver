% Computes the semi-analytical solution to the steady, linearized Euler equations
% in terrain following coordinates using a coordinate transformation from
% XZ to alpha-eta both from -pi to pi. The vertical boundary condition is
% also tranformed in the vertical so that infinity maps to eta = pi. The
% reference state is a standard atmosphere of piece-wise linear temperature
% profiles with a smooth zonal jet given. Pressure and density initialized
% in hydrostatic balance.

clc
clear
close all
opengl info;
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 1024; % Expansion order matches physical grid
NZ = 128; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;
iW = 1;
iP = 2;
iT = 3;

%% Set the test case and global parameters
%TestCase = 'ShearJetSchar'; BC = 2;
%TestCase = 'ShearJetScharCBVF'; BC = 2;
%TestCase = 'ClassicalSchar'; BC = 2;
TestCase = 'AndesMtn'; BC = 2;

z0 = 0.0;
gam = 1.4;
Rd = 287.06;
cp = 1004.5;
cv = cp - Rd;
ga = 9.80616;
p0 = 1.0E5;
kappa = Rd / cp;
if strcmp(TestCase,'ShearJetSchar') == true
    zH = 35000.0;
    l1 = - 1.0E4 * (3.0 * pi);
    l2 = 1.0E4 * (3.0 * pi);
    L = abs(l2 - l1);
    GAMT = -0.0065;
    HT = 11000.0;
    GAMS = 0.001;
    HML = 9000.0;
    HS = 20000.0;
    T0 = 300.0;
    BVF = 0.0;
    hfactor = 1.0;
    depth = 10000.0;
    width = 15000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = false;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    zH = 35000.0;
    l1 = -60000.0;
    l2 = 60000.0;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 300.0;
    BVF = 0.01;
    hfactor = 1.0;
    depth = 10000.0;
    width = 15000.0;
    nu1 = hfactor * 1.0 * 1.0E-2; nu2 = hfactor * 1.0 * 1.0E-2;
    nu3 = hfactor * 1.0 * 1.0E-2; nu4 = hfactor * 1.0 * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ClassicalSchar') == true
    zH = 35000.0;
    l1 = -60000.0;
    l2 = 60000.0;
    L = abs(l2 - l1);
    GAMT = 0.0;
    HT = 0.0;
    GAMS = 0.0;
    HML = 0.0;
    HS = 0.0;
    T0 = 300.0;
    BVF = 0.01;
    depth = 10000.0;
    width = 15000.0;
    hfactor = 1.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0 * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 10.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 0.0;
    b = 0.0;
elseif strcmp(TestCase,'AndesMtn') == true
    zH = 45000.0;
    l1 = - 1.0E5 * 4.0;
    l2 = 1.0E5 * 4.0;
    L = abs(l2 - l1);
    GAMT = -0.0065;
    HT = 11000.0;
    GAMS = 0.001;
    HML = 9000.0;
    HS = 20000.0;
    T0 = 300.0;
    BVF = 0.0;
    hfactor = 1.0;
    depth = 15000.0;
    width = 100000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = false;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 1000.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '100m';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
end

%% Set up physical parameters for basic state(taken from Tempest defaults)
BS = struct('gam',gam,'Rd',Rd,'cp',cp,'cv',cv,'GAMT',GAMT,'HT',HT,'GAMS', ...
            GAMS,'HML',HML,'HS',HS,'ga',ga,'p0',p0,'T0',T0,'BVF',BVF);

%% Set up the jet and mountain profile parameters
UJ = struct('u0',u0,'uj',uj,'b',b,'ga',ga);
DS = struct('z0',z0,'zH',zH,'l1',l1,'l2',l2,'L',L,'aC',aC,'lC',lC,'hC',hC,'hfilt',hfilt);

%% Set up the Rayleigh Layer with a coefficient one order of magnitude less than the order of the wave field
RAY = struct('depth',depth,'width',width,'nu1',nu1,'nu2',nu2,'nu3',nu3,'nu4',nu4);

%% Compute coarse and fine matrices and RHS vectors
tic;
[LD,FF,REFS] = ...
computeCoeffMatrixForceFFT(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);
[SOL,sysDex] = GetAdjust4CBC(REFS,BC,NX,NZ,OPS);
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = (FF - LD * SOL); clear LD FF;
% Solve the symmetric normal equations (in the coarsest grid)
%AN = A' * A;
%bN = A' * b(sysDex,1); clear A b;
AN = A;
bN = b(sysDex,1); clear A b;
%% Solve the coarse residual system by direct method
spparms('piv_tol', 1.0);
spparms('sym_tol', 1.0);
solc = AN \ bN;
SOL(sysDex) = solc; clear solc;

%% Unpack the coarse solution and interpolate up to fine
%
uxz = reshape(SOL((1:OPS)),NZ,NX);
wxz = reshape(SOL((1:OPS) + iW * OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + iP * OPS),NZ,NX);
txz = reshape(SOL((1:OPS) + iT * OPS),NZ,NX);
%%
save('coarseANbN', 'AN', 'bN', '-v7.3'); clear AN bN;
toc; disp('Direct solve on the coarsest mesh and save data... DONE!');

%% Compute a sequence of grids all the way to 100 m resolution
%
NX100 = round(L / 100, -1);
NZ100 = round(zH / 100, -1);

DNX = NX100 - NX;
DNZ = NZ100 - NZ;

%NXF = [2*NX 4*NX 8*NX];
%NZF = [2*NZ 350 450];
NXF = [1200 1400 1600];
NZF = [200 260 320];
OPSF = NXF .* NZF;
%}

%% Generate the fine grids and save the coefficient matrices
%
tic;
for nn=1:length(NXF)
    [LD,FF,REFSF(nn)] = ...
    computeCoeffMatrixForceFFT(DS, BS, UJ, RAY, TestCase, NXF(nn), NZF(nn), applyTopRL, applyLateralRL);

    [SOLF,sysDexF] = GetAdjust4CBC(REFSF(nn), BC, NXF(nn), NZF(nn), OPSF(nn));

    spparms('spumoni',2);
    A = LD(sysDexF,sysDexF);
    b = (FF - LD * SOLF); clear LD FF;
    AN = A;
    bN = b(sysDexF,1); clear A b;
    save(['fineANbN' int2str(OPSF(nn))], 'AN', 'bN', 'sysDexF', 'SOLF', '-v7.3'); clear AN bN;
end
toc; disp('Save fine meshes... DONE!');
%}
%% Solve up from the coarsest grid!
%
tic;
MITER = [400 200 100];
IITER = [40 20 10];
for nn=1:length(NXF)
    % IFFT to the fine mesh
    duxzint = real(ifft(uxz, NXF(nn), 2));
    dwxzint = real(ifft(wxz, NXF(nn), 2));
    dpxzint = real(ifft(pxz, NXF(nn), 2));
    dtxzint = real(ifft(txz, NXF(nn), 2));
    % Interpolate the vertical
    [zlc, ~] = chebdif(NZF(nn), 1);
    zlc = 0.5 * (zlc + 1.0);
    if nn == 1
        [duxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, duxzint, NXF(nn), NZF(nn), zlc);
        [dwxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, dwxzint, NXF(nn), NZF(nn), zlc);
        [dpxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, dpxzint, NXF(nn), NZF(nn), zlc);
        [dtxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, dtxzint, NXF(nn), NZF(nn), zlc);
    else
        [duxzint, ~, ~, ~] = VertLegInterp(REFSF(nn-1), DS, RAY, duxzint, NXF(nn), NZF(nn), zlc);
        [dwxzint, ~, ~, ~] = VertLegInterp(REFSF(nn-1), DS, RAY, dwxzint, NXF(nn), NZF(nn), zlc);
        [dpxzint, ~, ~, ~] = VertLegInterp(REFSF(nn-1), DS, RAY, dpxzint, NXF(nn), NZF(nn), zlc);
        [dtxzint, ~, ~, ~] = VertLegInterp(REFSF(nn-1), DS, RAY, dtxzint, NXF(nn), NZF(nn), zlc);
    end
    % FFT back to include the new k_x components (should have no power)
    duxzint = fft(duxzint, NXF(nn), 2);
    dwxzint = fft(dwxzint, NXF(nn), 2);
    dpxzint = fft(dpxzint, NXF(nn), 2);
    dtxzint = fft(dtxzint, NXF(nn), 2);

    dsol = [reshape(duxzint, OPSF(nn), 1); ...
            reshape(dpxzint, OPSF(nn), 1); ...
            reshape(dwxzint, OPSF(nn), 1); ...
            reshape(dtxzint, OPSF(nn), 1)];

    FP = load(['fineANbN' int2str(OPSF(nn))]);
    % Apply iterative solve up every level of mesh
    sol = gmres(FP.AN, FP.bN, IITER(nn), 1.0E-6, MITER(nn), [], [], dsol(FP.sysDexF));
    FP.SOLF(FP.sysDexF) = sol;
    
    % Compute the residual
    
    % Get the solution fields
    uxz = reshape(FP.SOLF((1:OPSF(nn))), NZF(nn), NXF(nn));
    wxz = reshape(FP.SOLF((1:OPSF(nn)) + iW * OPSF(nn)), NZF(nn), NXF(nn));
    pxz = reshape(FP.SOLF((1:OPSF(nn)) + iP * OPSF(nn)), NZF(nn), NXF(nn));
    txz = reshape(FP.SOLF((1:OPSF(nn)) + iT * OPSF(nn)), NZF(nn), NXF(nn));
    clear sol FP;
end
toc; disp('Run Full Multigrid Solution (N grids)... DONE!');
%}
%% Plot the solution frequency space
cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);
%
uxz = real(reshape(SOL((1:OPS)),NZ,NX));
wxz = real(reshape(SOL((1:OPS) + OPS),NZ,NX));
rxz = real(reshape(SOL((1:OPS) + 2*OPS),NZ,NX));
pxz = real(reshape(SOL((1:OPS) + 3*OPS),NZ,NX));
%
kdex = find(REFS.KF(1,:) >= 0.0);
rad2freq = 1. / (2. * pi);
wfreq = rad2freq * (REFS.KF(:,kdex));
wlen = wfreq.^(-1);
m2km = 1.0E-3;
im2ikm = 1.0E3;
%
mu = max(max(abs(uxz)));
mw = max(max(abs(wxz)));
mr = max(max(abs(rxz)));
mp = max(max(abs(pxz)));
xd = [1.0E-4 1.0E-1];
nc = 12;
%
figure;
subplot(2,1,1);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / mu * uxz(:,kdex),nc,'LineStyle','-'); grid on; caxis([-1.0 1.0]);
set(gca, 'XScale', 'log'); xlim(xd);
title('Horizontal Velocity U $(ms^{-1})$');
ylabel('Height (km)');

subplot(2,1,2);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / mw * wxz(:,kdex),nc,'LineStyle','-'); grid on; caxis([-1.0 1.0]);
%set(gca, 'XScale', 'log'); xlim(xd);
xlim([8.0E-2 1.0E-1]);
title('Vertical Velocity W $(ms^{-1})$');
xlabel('Spatial Frequency $(km^{-1})$');
ylabel('Height (km)');
drawnow
fname = ['FREQ_RESP01' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%
figure;
subplot(2,1,1);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / mr * rxz(:,kdex),nc,'LineStyle','-'); grid on; caxis([-1.0 1.0]);
set(gca, 'XScale', 'log'); xlim(xd);
title('$\ln \rho$');
ylabel('Height (km)');

subplot(2,1,2);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / mp * pxz(:,kdex),nc,'LineStyle','-'); grid on; caxis([-1.0 1.0]);
set(gca, 'XScale', 'log'); xlim(xd);
title('$\ln \theta$');
xlabel('Spatial Frequency $(km^{-1})$');
ylabel('Height (km)');
fname = ['FREQ_RESP02' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%pause;
%}
%% Plot the solution using the IFFT to recover the solution in physical space
%SOL(sysDex) = sol;
NXI = 4096;
uxz = real(ifft(reshape(SOL((1:OPS)),NZ,NX),NXI,2));
wxz = real(ifft(reshape(SOL((1:OPS) + OPS),NZ,NX),NXI,2));
rxz = real(ifft(reshape(SOL((1:OPS) + 2*OPS),NZ,NX),NXI,2));
pxz = real(ifft(reshape(SOL((1:OPS) + 3*OPS),NZ,NX),NXI,2));

uxz(:,end) = uxz(:,1);
wxz(:,end) = wxz(:,1);
rxz(:,end) = rxz(:,1);
pxz(:,end) = pxz(:,1);
%{
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,(REFS.ujref + uxz),21); colorbar; 
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U (m/s)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,wxz,21); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Vertical Velocity W (m/s)');
%
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,rxz,21); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Density (kg/m^3)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,pxz,21); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Pressure (Pa)');
drawnow
%}

%% Compute some of the fields needed for instability checks
lpt = REFS.lthref + pxz;
pt = exp(lpt);
lp = REFS.lpref + rxz;
p = exp(lp);
P = REFS.pref;
PT = REFS.thref;
rho = p ./ (Rd * pt) .* (p0 * p.^(-1)).^kappa;
R = p ./ (Rd * PT) .* (p0 * P.^(-1)).^kappa;
RT = (rho .* pt) - (R .* PT);

%% Compute Ri, Convective Parameter, and BVF
DDZ_BC = REFS.DDZ;
dlrho = REFS.dlrref + REFS.sigma .* (DDZ_BC * (log(rho) - REFS.lrref));
duj = REFS.dujref + REFS.sigma .* (DDZ_BC * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

DDZ_BC = REFS.DDZ;
dlpt = REFS.dlthref + REFS.sigma .* (DDZ_BC * real(pxz));
temp = p ./ (Rd * rho);
conv = temp .* dlpt;

RiREF = -BS.ga * REFS.dlrref(:,1);
RiREF = RiREF ./ (REFS.dujref(:,1).^2);

convREF = REFS.pref ./ (Rd * REFS.rref) .* REFS.dlthref;

xdex = 1:1:NX;
figure;
subplot(1,2,1); semilogx(Ri(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
semilogx(RiREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
ylabel('Height (km)');
title('Richardson Number');
xlim([0.1 1.0E4]);
ylim([0.0 25.0]);

subplot(1,2,2); 
plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * zH],'k--','LineWidth',2.5);
semilogx(convREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
%ylabel('Height (km)');
title('Convective Stability');
ylim([0.0 25.0]);
xlim([-0.06 0.06]);

fname = ['RI_CONV_N2_' TestCase num2str(hC)];
drawnow;
%screen2png(fname);
%% Compute N and the local Fr number
%
figure;
%DDZ_BC = REFS.DDZ;
%dlpres = REFS.dlpref + REFS.sigma .* (DDZ_BC * real(rxz));
NBVF = (ga .* dlpt);
NBVFREF = (ga .* REFS.dlthref);

Lv = hC;
FR = 2 * pi * abs(REFS.ujref + uxz) ./ (sqrt(NBVF) * Lv);
FRREF = 2 * pi * abs(REFS.ujref) ./ (sqrt(NBVFREF) * Lv);

xdex = 1:1:NX;
plot(FR(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
hold on;
semilogx([1.0 1.0],[0.0 1.0E5],'k--','LineWidth',2.5);
semilogx(FRREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
title('Local Froude Number');
xlabel('$Fr$');
ylabel('Height (km)');
ylim([0.0 25.0]);
xlim([-1.0 20.0]);
drawnow;

fname = ['FROUDE_' TestCase num2str(hC)];
%drawnow;
screen2png(fname);
%}

%% Interpolate to a regular grid using Hermite and Legendre transforms'
%{ 
%CHANGE THIS TO FOURIER INTERPOLATION!
NXI = 1600;
NZI = 200;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(pxz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
uxzint = uxz;
wxzint = wxz;
rxzint = rxz;
pxzint = pxz;
XI = m2km * REFS.XL;
ZI = m2km * REFS.ZTL;

XINT = XI;
ZINT = ZI;
%} 

%% INTERPOLATED GRID PLOTS
%
width = 0.0;
depth = 0.0;
% Compute the reference state initialization
%
if strcmp(TestCase,'ShearJetSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
elseif strcmp(TestCase,'ClassicalSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [ujref, ~] = computeJetProfileUniform(UJ, lpref);
elseif strcmp(TestCase,'AndesMtn') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
end
%}
figure;
colormap(cmap);
contourf(XI,ZI,uxzint,30); colorbar; grid on; cm = caxis;
%contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),2.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-200 300]);
%ylim([0 15]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('\textsf{$U^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Height (km)');
%screen2png(['UREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
contourf(XI,ZI,wxzint,30); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Height (km)');
%screen2png(['WREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
subplot(1,2,1); contourf(XI,ZI,rxzint,30); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln p)^{\prime} ~~ (Pa)$');
subplot(1,2,2); contourf(XI,ZI,pxzint,30); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln \theta)^{\prime} ~~ (K)$');
drawnow

%% Debug
%{
subplot(2,2,1); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS)),NZ,NX)); colorbar; xlim([-15000.0 15000.0]); ylim([0.0 1000.0]);
title('Total Horizontal Velocity U (m/s)');
subplot(2,2,2); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS) + OPS),NZ,NX)); colorbar; ylim([0.0 1000.0]);
title('Vertical Velocity W (m/s)');
subplot(2,2,3); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS) + 2*OPS),NZ,NX)); colorbar; ylim([0.0 1000.0]);
title('Perturbation Density (kg/m^3)');
subplot(2,2,4); surf(REFS.XL,REFS.ZTL,reshape(FBC((1:OPS) + 3*OPS),NZ,NX)); colorbar; ylim([0.0 1000.0]);
title('Perturbation Pressure (Pa)');
drawnow
%}

%% Save the data
%
close all
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceFFT' int2str(hC) '.mat'];
save(fileStore);
%}
