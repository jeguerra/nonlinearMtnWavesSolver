%% COMPUTES STEADY CONSERVATIVE FORM 2D MOUNTAIN WAVE PROBLEM IN 4 TEST CONFIGURATIONS:

% 1) 'ShearJetSchar' Discontinous background with strong shear
% 2) 'ShearJetScharCBVF' Uniform background with strong shear
% 3) 'ClassicalSchar' The typical Schar mountain test with uniform
% background and constant wind
% 4) 'AndesMtn' Same as 1) but with real input terrain data

clc
clear
close all
opengl info
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 96; % Expansion order matches physical grid
NZ = 128; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;
iW = 1;
iR = 2;
iT = 3;

%% Set the test case and global parameters
TestCase = 'ShearJetSchar'; BC = 1;
%TestCase = 'ShearJetScharCBVF'; BC = 1;
%TestCase = 'ClassicalSchar'; BC = 1;
%TestCase = 'AndesMtn'; BC = 1;

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
    width = 16000.0;
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
    hC = 100.0;
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
tic;
%% Compute the initialization and grid
[REFS, ~] = computeGridRefState_FluxForm(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[SOL,sysDex] = GetAdjust4CBC(REFS, BC, NX, NZ, OPS);

%% Compute the LHS coefficient matrix and force vector for the test case
[LD,FF] = ...
computeCoeffMatrixForce_FluxForm(BS, RAY, REFS);
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = (FF - LD * SOL); clear LD FF;
% Solve the symmetric normal equations (in the coarsest grid)
AN = A' * A;
bN = A' * b(sysDex,1); clear A b;
%% Solve the coarse residual system by direct method Cholesky decomposition
%solc = AN \ bN; clear AN;
[solc, cholParms] = cholmod2(AN, bN); clear AN;
disp(cholParms);
SOL(sysDex) = solc; clear solc;

%% Unpack the coarse solution and interpolate up to fine
druxz = reshape(SOL((1:OPS)),NZ,NX);
drwxz = reshape(SOL((1:OPS) + iW * OPS),NZ,NX);
drxz = reshape(SOL((1:OPS) + iR * OPS),NZ,NX);
drtxz = reshape(SOL((1:OPS) + iT * OPS),NZ,NX);
%%
toc; disp('Direct solve on the coarsest mesh and save data... DONE!');

%% Compute a sequence of grids all the way to 100 m resolution
NX100 = round(L / 100);
NZ100 = round(zH / 100);

DNX = NX100 - NX;
DNZ = NZ100 - NZ;

NXF = [192 256 384 512];
NZF = [192 256 360 360];
OPSF = NXF .* NZF;

%% Generate the fine grids and save the coefficient matrices
tic;
for nn=1:length(NXF)
    [REFSF(nn), DOPS(nn)] = computeGridRefState_FluxForm(DS, BS, UJ, RAY, TestCase, NXF(nn), NZF(nn), applyTopRL, applyLateralRL);
    DOPSF = DOPS(nn);
    [SOLF,sysDexF] = GetAdjust4CBC(REFSF(nn), BC, NXF(nn), NZF(nn), OPSF(nn));

    spparms('spumoni',2);
    b = computeCoeffMatrixMulFluxForm(REFSF(nn), DOPS(nn), SOLF, []);
    bN = - b(sysDexF,1); clear b;
    save(['fineANbN' int2str(OPSF(nn))], 'DOPSF', 'bN', 'sysDexF', 'SOLF', '-v7.3');
    %save('-binary', ['fineANbN' int2str(OPSF(nn))], 'bN', 'sysDexF', 'SOLF'); clear AN bN;
end
toc; disp('Save fine meshes... DONE!');

%% Solve up from the coarsest grid!
tic;
MITER = [400 200 100];
IITER = [40 20 10];
for nn=1:length(NXF)
    
    if nn == 1
        FP = load(['fineANbN' int2str(OPSF(nn))]);
        [xh,~] = herdif(NXF(nn), 1, 0.5*L, true);
        [zlc, ~] = chebdif(NZF(nn), 1);
        zlc = 0.5 * (zlc + 1.0);
        [druxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(druxz), NXF(nn), NZF(nn), xh, zlc);
        [drwxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(drwxz), NXF(nn), NZF(nn), xh, zlc);
        [drxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(drxz), NXF(nn), NZF(nn), xh, zlc);
        [drtxzint, ~, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(drtxz), NXF(nn), NZF(nn), xh, zlc);
        clear duxz dwxz dpxz dtxz;
    else
        FP = load(['fineANbN' int2str(OPSF(nn))]);
        [xh,~] = herdif(NXF(nn), 1, 0.5*L, true);
        [zlc, ~] = chebdif(NZF(nn), 1);
        zlc = 0.5 * (zlc + 1.0);
        [druxzint, ~, ~, ~] = HerTransLegInterp(REFSF(nn-1), DS, RAY, real(ruxz), NXF(nn), NZF(nn), xh, zlc);
        [drwxzint, ~, ~, ~] = HerTransLegInterp(REFSF(nn-1), DS, RAY, real(rwxz), NXF(nn), NZF(nn), xh, zlc);
        [drxzint, ~, ~, ~] = HerTransLegInterp(REFSF(nn-1), DS, RAY, real(rxz), NXF(nn), NZF(nn), xh, zlc);
        [drtxzint, ~, ~, ~] = HerTransLegInterp(REFSF(nn-1), DS, RAY, real(rtxz), NXF(nn), NZF(nn), xh, zlc);
        %clear uxz wxz pxz txz;
    end

    dsol = [reshape(druxzint, OPSF(nn), 1); ...
            reshape(drwxzint, OPSF(nn), 1); ...
            reshape(drxzint, OPSF(nn), 1); ...
            reshape(drtxzint, OPSF(nn), 1)];

    % Apply iterative solve up every level of mesh
    sol = gmres(FP.AN, FP.bN, IITER(nn), 1.0E-6, MITER(nn), [], [], dsol(FP.sysDexF));
    FP.SOLF(FP.sysDexF) = sol;
    
    % Compute the residual
    
    % Get the solution fields
    ruxz = reshape(FP.SOLF((1:OPSF(nn))), NZF(nn), NXF(nn));
    rwxz = reshape(FP.SOLF((1:OPSF(nn)) + iW * OPSF(nn)), NZF(nn), NXF(nn));
    rxz = reshape(FP.SOLF((1:OPSF(nn)) + iR * OPSF(nn)), NZF(nn), NXF(nn));
    rtxz = reshape(FP.SOLF((1:OPSF(nn)) + iT * OPSF(nn)), NZF(nn), NXF(nn));
    clear sol FP;
end
toc; disp('Run Full Multigrid Solution (N grids)... DONE!');

%% Compute the kinematic fields from (u, w, rho, rho-theta)
uxz = (REFSF(nn).rref .* REFSF(nn).ujref + ruxz) ./ (REFSF(nn).rref + rxz) - REFSF(nn).ujref;
wxz = rwxz ./ (REFSF(nn).rref + rxz);
txz = (REFSF(nn).rref .* REFSF(nn).thref + rtxz) ./ (REFSF(nn).rref + rxz) - REFSF(nn).thref;

%% Interpolate to a regular grid using Hermite and Legendre transforms'
%
NXI = 3001;
NZI = 351;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFSF(nn), DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFSF(nn), DS, RAY, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFSF(nn), DS, RAY, real(rxz), NXI, NZI, 0, 0);
[txzint, ~, ~] = HerTransLegInterp(REFSF(nn), DS, RAY, real(txz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}

% Plot the solution in the native grids
%{
% NATIVE GRID PLOTS
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(REFS.ujref + uxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U (m/s)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(wxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Vertical Velocity W (m/s)');

fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(1,2,1); contourf(REFS.XL,REFS.ZTL,real(rxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Density (kg/m^3)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(pxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Log Pressure (Pa)');
drawnow
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
dlthref = 1.0 / BS.gam * dlpref - dlrref;

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
drawnow;
export_fig(['UREferenceSolution' mtnh '.png']);
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
drawnow;
export_fig(['WREferenceSolution' mtnh '.png']);
%
figure;
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,rxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$\rho^{\prime} ~~ (kgm^{-3})$');
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,txzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$\theta^{\prime} ~~ (K)$');
drawnow
%
%% Compute some of the fields needed for instability checks
pt = REFSF(nn).thref + txz;
rho = REFSF(nn).rref + rxz;
P = REFSF(nn).pref;
PT = REFSF(nn).thref;
p = ((Rd * rho .* pt) * (p0^(-kappa))).^(kappa - 1.0); 
R = rho;
RT = rtxz;

%% Compute Ri, Convective Parameter, and BVF
DDZ_BC = REFSF(nn).DDZ;
dlrho = REFSF(nn).dlrref + REFSF(nn).sigma .* (DDZ_BC * (log(rho) - REFSF(nn).lrref));
duj = REFSF(nn).dujref + REFSF(nn).sigma .* (DDZ_BC * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

DDZ_BC = REFSF(nn).DDZ;
dlpt = (REFSF(nn).dthref + REFSF(nn).sigma .* (DDZ_BC * real(txz))) .* (pt.^(-1));
temp = (P + p) ./ (Rd * rho);
conv = temp .* dlpt;

RiREF = -BS.ga * REFSF(nn).dlrref(:,1);
RiREF = RiREF ./ (REFSF(nn).dujref(:,1).^2);

xdex = 1:1:NX;
figure;
subplot(1,2,1); semilogx(Ri(:,xdex),1.0E-3*REFSF(nn).ZTL(:,xdex),'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
semilogx(RiREF,1.0E-3*REFSF(nn).ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
ylabel('Elevation (km)');
title('Richardson Number');
xlim([0.1 1.0E4]);
ylim([0.0 30.0]);

subplot(1,2,2); plot(conv(:,xdex),1.0E-3*REFSF(nn).ZTL(:,xdex),'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * zH],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
%ylabel('Elevation (km)');
title('Convective Stability');
%xlim([-0.3 0.3]);

fname = ['RI_CONV_N2_' TestCase num2str(hC) '.png'];
drawnow;
export_fig(fname);
%% Compute N and the local Fr number
%{
figure;
DDZ_BC = REFS.DDZ;
dlpres = REFS.dlpref + REFS.sigma .* (DDZ_BC * log(p));
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
subplot(2,2,1); surf(XI,ZI,uxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('U (m/s)');
subplot(2,2,2); surf(XI,ZI,wxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('W (m/s)');
subplot(2,2,3); surf(XI,ZI,exp(lrref) .* (exp(rxzint) - 1.0)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('$(\ln p)^{\prime}$ (Pa)');
subplot(2,2,4); surf(XI,ZI,exp(lpref) .* (exp(pxzint) - 1.0)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('$(\ln \theta)^{\prime}$ (K)');
drawnow
%}

%% Save the data
%{
close all;
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER_FluxForm' char(TestCase) int2str(hC) '.mat'];
save(fileStore);
%}