
%% COMPUTES STEADY LNP-LNT 2D MOUNTAIN WAVE PROBLEM IN 4 TEST CONFIGURATIONS:

% 1) 'ShearJetSchar' Discontinous background with strong shear
% 2) 'ShearJetScharCBVF' Uniform background with strong shear
% 3) 'ClassicalSchar' The typical Schar mountain test with uniform
% background and constant wind
% 4) 'AndesMtn' Same as 1) but with real input terrain data

%clc
clear
close all
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 80; % Expansion order matches physical grid
NZ = 100; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;
iW = 2;
iP = 1;
iT = 3;

%% Set the test case and global parameters
TestCase = 'ShearJetSchar'; BC = 0;
%TestCase = 'ShearJetScharCBVF'; BC = 0;
%TestCase = 'ClassicalSchar'; BC = 0;
%TestCase = 'AndesMtn'; BC = 0;

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
    l1 = -1.0E4 * 2.0 * pi;
    l2 = 1.0E4 * 2.0 * pi;
    %l1 = -6.0E4;
    %l2 = 6.0E4;
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
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 100.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    zH = 35000.0;
    %l1 = -1.0E4 * 2.0 * pi;
    %l2 = 1.0E4 * 2.0 * pi;
    l1 = -6.0E4;
    l2 = 6.0E4;
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
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
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
    %l1 = -1.0E4 * 2.0 * pi;
    %l2 = 1.0E4 * 2.0 * pi;
    l1 = -6.0E4;
    l2 = 6.0E4;
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
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
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
    zH = 40000.0;
    l1 = -1.0E5 * 2.0 * pi;
    l2 = 1.0E5 * 2.0 * pi;
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
    width = 101000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 100.0;
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
[LD,FF,REFS] = ...
computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);
[SOL,sysDex] = GetAdjust4CBC(REFS,BC,NX,NZ,OPS);
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = (FF - LD * SOL); clear LD FF;
% Solve the symmetric normal equations (in the coarsest grid)
AN = A' * A;
bN = A' * b(sysDex,1); clear A b;
save('coarseANbN', 'AN', 'bN', '-v7.3'); clear AN bN;

%% Compute a sequence of grids all the way to 100 m resolution
NXF = NX + 120;
NZF = NZ + 240;
OPSF = NXF * NZF;

[LD,FF,REFSF] = ...
computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NXF, NZF, applyTopRL, applyLateralRL);

[SOLF,sysDexF] = GetAdjust4CBC(REFSF,BC,NXF,NZF,OPSF);

spparms('spumoni',2);
A = LD(sysDexF,sysDexF);
b = (FF - LD * SOLF); clear LD FF;
AN = A;
bN = b(sysDexF,1); clear A b;
save('fineANbN', 'AN', 'bN', '-v7.3'); clear AN bN;
%
%% Apply 2 iterations of LSQR to the  fine problem
%{
FP = load('fineANbN');
sol = lsqr(FP.AN, 0.0*FP.bN, 0.1, 2);
SOLF(sysDexF) = sol;
rsl = zeros(size(SOLF));
rsl(sysDexF) = FP.bN - FP.AN * sol; clear FP sol;

%% Unpack the fine solution and interpolate down to coarse
ruxz = reshape(rsl((1:OPSF)),NZF,NXF);
rwxz = reshape(rsl((1:OPSF) + iW * OPSF),NZF,NXF);
rpxz = reshape(rsl((1:OPSF) + iP * OPSF),NZF,NXF);
rtxz = reshape(rsl((1:OPSF) + iT * OPSF),NZF,NXF);

[xh,~] = herdif(NX, 1, 0.5*L, true);
[zlc, ~] = chebdif(NZ, 1);
zlc = 0.5 * (zlc + 1.0);
[ruxzint, ~, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(ruxz), NX, NZ, xh, zlc);
[rwxzint, ~, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(rwxz), NX, NZ, xh, zlc);
[rpxzint, ~, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(rpxz), NX, NZ, xh, zlc);
[rtxzint, ~, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(rtxz), NX, NZ, xh, zlc);

rslc = [reshape(ruxzint, OPS, 1); ...
        reshape(rpxzint, OPS, 1); ...
        reshape(rwxzint, OPS, 1); ...
        reshape(rtxzint, OPS, 1)];
rslc = rslc(sysDex);
%}
%% Solve the coarse residual system by direct method
CP = load('coarseANbN');
solc = CP.AN \ CP.bN; clear CP;
SOL(sysDex) = solc; clear solc;

%% Unpack the coarse solution and interpolate up to fine
duxz = reshape(SOL((1:OPS)),NZ,NX);
dwxz = reshape(SOL((1:OPS) + iW * OPS),NZ,NX);
dpxz = reshape(SOL((1:OPS) + iP * OPS),NZ,NX);
dtxz = reshape(SOL((1:OPS) + iT * OPS),NZ,NX);

[xh,~] = herdif(NXF, 1, 0.5*L, true);
[zlc, ~] = chebdif(NZF, 1);
zlc = 0.5 * (zlc + 1.0);
[duxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(duxz), NXF, NZF, xh, zlc);
[dwxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(dwxz), NXF, NZF, xh, zlc);
[dpxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(dpxz), NXF, NZF, xh, zlc);
[dtxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(dtxz), NXF, NZF, xh, zlc);

%dsol = [reshape(duxzint, OPSF, 1); reshape(dwxzint, OPSF, 1); reshape(dpxzint, OPSF, 1); reshape(dtxzint, OPSF, 1)];
dsol = [reshape(duxzint, OPSF, 1); ...
        reshape(dpxzint, OPSF, 1); ...
        reshape(dwxzint, OPSF, 1); ...
        reshape(dtxzint, OPSF, 1)];
clear duxz dwxz dpxz dtxz;
SOLF(sysDexF) = SOLF(sysDexF) + dsol(sysDexF);

%% Apply a final set of iterations on the fine problem
FP = load('fineANbN');
setup.type = 'crout';
setup.milu = 'row';
setup.droptol = 1.0E-2;
[L,U] = ilu(FP.AN ,setup);
sol = gmres(FP.AN, FP.bN, 5, 1.0E-6, 100, L, U, SOLF(sysDexF)); clear L U;
%sol = gmres(FP.AN, FP.bN, 5, 1.0E-6, 20, [], [], SOLF(sysDexF));
%sol = lsqr(FP.AN, FP.bN, 1.0E-6, 100, [], [], SOLF(sysDexF));
%sol = cgs(FP.AN, FP.bN, 1.0E-6, 100, [], [], SOLF(sysDexF));
SOLF(sysDexF) = sol; clear sol FP;

%% Get the solution fields
uxz = reshape(SOLF((1:OPSF)),NZF,NXF);
wxz = reshape(SOLF((1:OPSF) + iW * OPSF),NZF,NXF);
pxz = reshape(SOLF((1:OPSF) + iP * OPSF),NZF,NXF);
txz = reshape(SOLF((1:OPSF) + iT * OPSF),NZF,NXF);

% Plot the solution in the native grids
%{
% NATIVE GRID PLOTS
figure;
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(REFS.ujref + uxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('\textsf{$U^{\prime} ~~ (ms^{-1})$}');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(wxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');

figure;
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(rxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('$(\ln p)^{\prime} ~~ (Pa)$');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(pxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('$(\ln \theta)^{\prime} ~~ (K)$');
drawnow
%}

%% Interpolate to a nice regular grid using Hermite and Legendre transforms'
%
NXI = 3001;
NZI = 351;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFSF, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(pxz), NXI, NZI, 0, 0);
[txzint, ~, ~] = HerTransLegInterp(REFSF, DS, RAY, real(txz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}

% Use the NCL hotcold colormap
cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

% INTERPOLATED GRID PLOTS
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
%
figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,uxzint,31); colorbar; grid on; cm = caxis;
%contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-200 300]);
%ylim([0 15]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('\textsf{$U^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['UREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['WREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,pxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln p)^{\prime} ~~ (Pa)$');
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,txzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln \theta)^{\prime} ~~ (K)$');
drawnow
%
%% Compute some of the fields needed for instability checks
lpt = REFSF.lthref + txz;
pt = exp(lpt);
lp = REFSF.lpref + pxz;
p = exp(lp);
P = REFSF.pref;
PT = REFSF.thref;
rho = p ./ (Rd * pt) .* (p0 * p.^(-1)).^kappa;
R = p ./ (Rd * PT) .* (p0 * P.^(-1)).^kappa;
RT = (rho .* pt) - (R .* PT);

%% Compute Ri, Convective Parameter, and BVF
DDZ_BC = REFSF.DDZ;
dlrho = REFSF.dlrref + REFSF.sigma .* (DDZ_BC * (log(rho) - REFSF.lrref));
duj = REFSF.dujref + REFSF.sigma .* (DDZ_BC * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

DDZ_BC = REFSF.DDZ;
dlpt = REFSF.dlthref + REFSF.sigma .* (DDZ_BC * real(txz));
temp = p ./ (Rd * rho);
conv = temp .* dlpt;

RiREF = -BS.ga * REFSF.dlrref(:,1);
RiREF = RiREF ./ (REFSF.dujref(:,1).^2);

xdex = 1:1:NX;
figure;
subplot(1,2,1); semilogx(Ri(:,xdex),1.0E-3*REFSF.ZTL(:,xdex),'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
semilogx(RiREF,1.0E-3*REFSF.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
ylabel('Elevation (km)');
title('Richardson Number');
xlim([0.1 1.0E4]);
ylim([0.0 30.0]);

subplot(1,2,2); plot(conv(:,xdex),1.0E-3*REFSF.ZTL(:,xdex),'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * zH],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
%ylabel('Elevation (km)');
title('Convective Stability');
%xlim([-0.3 0.3]);

fname = ['RI_CONV_N2_' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%% Compute N and the local Fr number
%{
fig = figure('Position',[0 0 2000 1000]); fig.Color = 'w';
DDZ_BC = REFS.DDZ;
dlpres = REFS.dlpref + REFS.sigma .* (DDZ_BC * real(pxz));
NBVF = (ga .* dlpt);

Lv = 2.5E3;
FR = 2 * pi * abs(REFS.ujref + uxz) ./ (sqrt(NBVF) * Lv);

xdex = 1:1:NX;
plot(FR(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on; grid minor;
title('Local Froude Number');
xlabel('$Fr$');
%ylabel('\textsf{Altitude (km)}','Interpreter','latex');
ylim([0.0 25.0]);
%xlim([-1.0E-3 2.0E-3]);
drawnow;

fname = ['FROUDE_' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%}
%% Debug
%{
figure;
subplot(2,2,1); surf(XI,ZI,uxzint); colorbar; xlim([-10000.0 25000.0]); ylim([0.0 2000.0]);
title('U (m/s)'); %zlim([-0.1 0.1]);
subplot(2,2,2); surf(XI,ZI,wxzint); colorbar; xlim([-10000.0 25000.0]); ylim([0.0 2000.0]);
title('W (m/s)');
subplot(2,2,3); surf(XI,ZI,exp(lrref) .* (exp(rxzint) - 1.0)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 2000.0]);
title('$(\ln p)^{\prime}$ (Pa)');
subplot(2,2,4); surf(XI,ZI,exp(lpref) .* (exp(pxzint) - 1.0)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 2000.0]);
title('$(\ln \theta)^{\prime}$ (K)');
drawnow
%}

%% Plot the terrain forcing
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
%{
close all;
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER_LnP' char(TestCase) int2str(hC) '.mat'];
save(fileStore);
%}
