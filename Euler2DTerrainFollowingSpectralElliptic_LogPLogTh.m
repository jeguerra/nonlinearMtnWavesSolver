
%% COMPUTES STEADY LNP-LNT 2D MOUNTAIN WAVE PROBLEM IN 4 TEST CONFIGURATIONS:
%% LIMITED RESOLUTION DIRECT SOLVER: UMFPACK LU
%% USED TO TEST ITERATIVE SOLVERS: GMRES AND ALSQR

% 1) 'ShearJetSchar' Discontinous background with strong shear
% 2) 'ShearJetScharCBVF' Uniform background with strong shear
% 3) 'ClassicalSchar' The typical Schar mountain test with uniform
% background and constant wind
% 4) 'AndesMtn' Same as 1) but with real input terrain data

clc
clear
close all
opengl info
addpath(genpath('/home/jeguerra/Documents/MATLAB/'));
startup;
warning('off');

%% Create the dimensional XZ grid
NX = 128; % Expansion order matches physical grid
NZ = 84; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;
iW = 1;
iP = 2;
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
    zH = 36000.0;
    l1 = -1.0E4 * 3.0 * pi;
    l2 = 1.0E4 * 3.0 * pi;
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
    width = 20000.0;
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
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    zH = 35000.0;
    l1 = -1.0E4 * 2.0 * pi;
    l2 = 1.0E4 * 2.0 * pi;
    %l1 = -6.0E4;
    %l2 = 6.0E4;
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
    hC = 1000.0;
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

%% Compute the initialization and grid
[REFS, DOPS] = computeGridRefState_LogPLogTh(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[SOL,sysDex] = GetAdjust4CBC(REFS, BC, NX, NZ, OPS);

%% Compute the LHS coefficient matrix and force vector for the test case
%[LD, FF] = ...
%computeCoeffMatrixForce_LogPLogTh(BS, RAY, REFS);
ZSPR = sparse(OPS,OPS);
LD = [DOPS.LD11 DOPS.LD12 DOPS.LD13 ZSPR;      ...
      ZSPR      DOPS.LD22 DOPS.LD23 DOPS.LD24; ...
      DOPS.LD31 DOPS.LD32 DOPS.LD33 ZSPR;      ...
      ZSPR      DOPS.LD42 ZSPR      DOPS.LD44];

%% Compute coupled multipoint BC by adjusting columns of LD
ubdex = 1:NZ:(OPS - NZ + 1);
wbdex = ubdex + iW*OPS;
dhdx = spdiags((REFS.DZT(1,:))', 0, NX, NX);
% Apply column adjustment for the multipoint coupled BC
LD(:,ubdex) = LD(:,ubdex) + LD(:,wbdex) * dhdx;
% Compute RHS scaling
WBC = REFS.DZT(1,:) .* REFS.ujref(1,:);

%% Solve the system by letting matlab \ do its thing...
%
disp('Solve by direct Cholesky coarse and ALSQR fine.');
tic
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = -LD(:,wbdex) * WBC'; clear LD FF;
figure; plot(b);
% Solve the symmetric normal equations
AN = A' * A;
bN = A' * b(sysDex); clear A b;
toc; disp('Compute coarse coefficient matrix... DONE!');
%sol = umfpack(AN, '\', bN); clear AN bN;
%sol = cholmod2(AN, bN); clear AN bN;
sol = AN \ bN; clear AN bN
toc; disp('Solve the first system by \... DONE!');
%pause;

%% Get the solution fields
SOL(sysDex) = sol;
SOL(wbdex) = REFS.DZT(1,:)' .* (REFS.ujref(1,:)' + SOL(ubdex));
%clear sol;
uxz = reshape(SOL((1:OPS)),NZ,NX);
wxz = reshape(SOL((1:OPS) + iW * OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + iP * OPS),NZ,NX);
txz = reshape(SOL((1:OPS) + iT * OPS),NZ,NX);

%% Here we can turn off the fine grid extension
%{
%% Use the coarse system solution to accelerate iterative solution of a finer system
NX = 300;
NZ = 140;
OPS = NX * NZ;

%% Compute the initialization and grid
[REFSI, DOPS] = computeGridRefState_LogPLogTh(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[SOL,sysDex] = GetAdjust4CBC(REFSI, BC, NX, NZ, OPS);

[xh,~] = herdif(NX, 1, 0.5*L, true);
[zlc, ~] = chebdif(NZ, 1);
zlc = 0.5 * (zlc + 1.0);
[uxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NX, NZ, xh, zlc);
[wxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NX, NZ, xh, zlc);
[pxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(pxz), NX, NZ, xh, zlc);
[txzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(txz), NX, NZ, xh, zlc);
REFS = REFSI;

xsol = [reshape(uxzint, OPS, 1); reshape(wxzint, OPS, 1); reshape(pxzint, OPS, 1); reshape(txzint, OPS, 1)];
toc; disp('Compute coase/fine interpolation... DONE!');

%% Compute the LHS coefficient matrix and force vector for the test case
tic;
spparms('spumoni',2);

%% Compute coupled multipoint BC by adjusting columns of LD
ZSPR = sparse(OPS,OPS);
LD = [DOPS.LD11 DOPS.LD12 DOPS.LD13 ZSPR;      ...
      ZSPR      DOPS.LD22 DOPS.LD23 DOPS.LD24; ...
      DOPS.LD31 DOPS.LD32 DOPS.LD33 ZSPR;      ...
      ZSPR      DOPS.LD42 ZSPR      DOPS.LD44];
ubdex = 1:NZ:(OPS - NZ + 1);
wbdex = ubdex + iW*OPS;
dhdx = spdiags((REFSI.DZT(1,:))', 0, NX, NX);
% Apply column adjustment for the multipoint coupled BC
LD(:,ubdex) = LD(:,ubdex) + LD(:,wbdex) * dhdx;
% Compute RHS scaling
SOL(wbdex) = REFSI.DZT(1,:) .* REFSI.ujref(1,:);

b = computeCoeffMatrixMulLogPLogTh(REFSI, DOPS, SOL, []); clear DOPS;
AN = LD(sysDex, sysDex);
bN = - b(sysDex,1); clear b;
toc; disp('Compute fine RHS... DONE!');
tic;
matMul = @(xVec) computeCoeffMatrixMulLogPLogTh(REFSI, DOPS, xVec, sysDex);
% Use GMRES to make the initial solution
[sol, ~, ~, ~, rvc1] = gmres(AN, bN, 10, 1.0E-6, 10, [], [], xsol(sysDex));
% Use ALSQR iterative method preconditioned with GMRES
matMul1 = @(xVec, ntrans) computeAorATMulLogPLogTh(REFSI, DOPS, xVec, sysDex, ntrans);
OPTS.LTOL = 1.0E-8;
OPTS.K = 10;
OPTS.MAXITL = 20;
OPTS.MAXITP = 20;
[sol,  ~, rvc3] = alsqr(AN, bN, sol, OPTS);

toc; disp('Solve by using iterative method with coarser initial guess... DONE!');

%% Get the solution fields
SOL(sysDex) = sol;
clear sol;
uxz = reshape(SOL((1:OPS)),NZ,NX);
wxz = reshape(SOL((1:OPS) + OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + 2*OPS),NZ,NX);
txz = reshape(SOL((1:OPS) + 3*OPS),NZ,NX);
%}
%% Plot the solution in the native grids
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
NZI = 451;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(pxz), NXI, NZI, 0, 0);
[txzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(txz), NXI, NZI, 0, 0);

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
%%
figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,uxzint,31); colorbar; grid on; cm = caxis;
%contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('\textsf{$U^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Height (km)');
%export_fig(['UREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Height (km)');
%export_fig(['WREferenceSolution' mtnh '.png']);
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
lpt = REFS.lthref + txz;
pt = exp(lpt);
lp = REFS.lpref + pxz;
p = exp(lp);
P = REFS.pref;
PT = REFS.thref;
rho = p ./ (Rd * pt) .* (p0 * p.^(-1)).^kappa;
R = p ./ (Rd * PT) .* (p0 * P.^(-1)).^kappa;
RT = (rho .* pt) - (R .* PT);

%% Compute Ri, Convective Parameter, and BVF
DDZ_BC = REFS.DDZ_L;
dlrho = REFS.dlrref + REFS.sigma .* (DDZ_BC * (log(rho) - REFS.lrref));
duj = REFS.dujref + REFS.sigma .* (DDZ_BC * real(uxz));

dlpt = REFS.dlthref + REFS.sigma .* (DDZ_BC * real(txz));
temp = p ./ (Rd * rho);
conv = temp .* dlpt;

Ri = ga * dlpt ./ (duj.^2);
RiREF = BS.ga * REFS.dlthref(:,1);
RiREF = RiREF ./ (REFS.dujref(:,1).^2);

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
ylim([0.0 30.0]);

subplot(1,2,2); plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * zH],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
%ylabel('Height (km)');
title('Convective Stability');
xlim([-0.01 0.025]);
ylim([0.0 30.0]);

fname = ['RI_CONV_N2_' TestCase num2str(hC)];
drawnow;
export_fig(fname);

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
