% Computes the semi-analytical solution to the steady, linearized Euler equations
% in terrain following coordinates using a coordinate transformation from
% XZ to alpha-eta both from -pi to pi. The vertical boundary condition is
% also tranformed in the vertical so that infinity maps to eta = pi. The
% reference state is a standard atmosphere of piece-wise linear temperature
% profiles with a smooth zonal jet given. Pressure and density initialized
% in hydrostatic balance.

clc
clear
%close all
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 100; % Expansion order matches physical grid
NXO = 80; % Expansion order
NZ = 100; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;

%% Set the test case and global parameters
%TestCase = 'ShearJetSchar'; BC = 0;
%TestCase = 'ShearJetScharCBVF'; BC = 0;
TestCase = 'ClassicalSchar'; BC = 0;
%TestCase = 'AndesMtn'; BC = 0;

z0 = 0.0;
gam = 1.4;
Rd = 287.06;
cp = 1004.5;
cv = cp - Rd;
ga = 9.80616;
p0 = 1.0E5;
if strcmp(TestCase,'ShearJetSchar') == true
    zH = 35000.0;
    %l1 = -1.0E4 * 2.0 * pi;
    %l2 = 1.0E4 * 2.0 * pi;
    l1 = -6.0E4;
    l2 = 6.0E4;
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
    hC = 10.0;
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

%% Compute the LHS coefficient matrix and force vector for the test case
[LD,FF,REFS] = ...
computeCoeffMatrixForceCBC(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL);

%% Get the boundary conditions
[FFBC,SOL,sysDex] = GetAdjust4CBC(BC,NX,NZ,OPS,FF);

%% Solve the system by projecting to orthogonal space first then solving
%{
%parpool('local');
disp('Solve by applying orthogonal basis and matlab \.');
tic
spparms('spumoni',2);
A = LD(sysDex,sysDex);
b = FFBC(sysDex,1);
toc; disp('Compute raw coefficient matrix... DONE.');
% Project onto the orthogonal subspace in the range of A
tic
AO = sporth(A);
toc; disp('Compute orthonormal basis... DONE.');
tic
AR = full(AO') * full(A);
AR = sparse(AR * full(AO));
BR = AO' * b;
toc; disp('Compute projection to orthogonal basis... DONE.');
clear LD FF FFBC;
tic
sol = AO * (AR \ BR);
toc; disp('Solve the system... DONE.');
%}
%% Solve the system by letting matlab \ do its thing...
%
disp('Solve by using matlab \ only.');
tic
spparms('spumoni',2);
%A = LD(sysDex,sysDex);
%b = FFBC(sysDex,1);
%disp(size(A));
% Normal equations...
A = LD(sysDex,sysDex)' * LD(sysDex,sysDex);
b = LD(sysDex,sysDex)' * FFBC(sysDex,1);
disp(size(A));
toc; disp('Compute raw coefficient matrix... DONE.');
clear LD FF FFBC;
tic
sol = (A \ b);
toc; disp('Solve the system... DONE.');
%tic
%[dvecs, dlambda] = eigs(LD(sysDex,sysDex),20,'lr');
%disp(diag(dlambda));
%clear LD
%toc; disp('Compute some eigenvalues/vectors... DONE.');
%}
clear A b
%% Get the solution fields
SOL(sysDex) = sol(:,1);
clear sol;
%%
uxz = reshape(SOL((1:OPS)),NZ,NX);
wxz = reshape(SOL((1:OPS) + OPS),NZ,NX);
rxz = reshape(SOL((1:OPS) + 2*OPS),NZ,NX);
pxz = reshape(SOL((1:OPS) + 3*OPS),NZ,NX);

%% Interpolate to a regular grid using Hermite and Legendre transforms'
%
NXI = 1000;
NZI = 200;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(pxz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
%% Interpolate to a regular grid using Hermite and Laguerre transforms'
%{
NXI = 1000;
NZI = 200;
[uxzint, XINT, ZINT, ZLINT] = HerTransLagTrans(REFS, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLagTrans(REFS, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLagTrans(REFS, DS, RAY, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLagTrans(REFS, DS, RAY, real(pxz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
% Plot the solution in the native grids
%
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

% Use the NCL hotcold colormap
cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

% INTERPOLATED GRID PLOTS
% Compute the reference state initialization
%
if strcmp(TestCase,'ShearJetSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
    %[lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT, RAY);
    %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
    %[lprefU,~,dlprefU,~] = computeBackgroundPressureCBVF(BS, ZLINT);
    %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ClassicalSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [ujref, ~] = computeJetProfileUniform(UJ, lpref);
elseif strcmp(TestCase,'AndesMtn') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
    %[lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT, RAY);
    %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
end
%}

figure;
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,uxzint,31); colorbar; grid on; cm = caxis;
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
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,rxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln p)^{\prime} ~~ (Pa)$');
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,pxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln \theta)^{\prime} ~~ (K)$');
drawnow
%}
%{
%% Compare W scatter plot to the predicted growth rate
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);

Im = REFS.dlthref(:,1) + 0.5 * REFS.dujref(:,1) ./ REFS.ujref(:,1) - 0.5 * REFS.dlpref(:,1);
MZ = exp(REFS.sigma(:,1) .* Im);
for xx=2:length(MZ)
    MZ(xx) = MZ(xx) * MZ(xx-1);
end
% Normalize and scale to the W field
MZ = max(max(wxzint)) / max(MZ) * MZ; 

plot(wxzint,1.0E-3 * ZI,'ks','LineWidth',1.5); grid on; hold on;
plot(MZ, 1.0E-3 * REFS.ZTL(:,1),'r-s','LineWidth',1.5); hold off;
%hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
%xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
%caxis([-0.08 0.08]);
title('Vertical Velocity Growth W $(m~s^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['WaveGrowthW_LnP' mtnh '.png']);
%}
%
%% Compute the scaling constants needed for residual diffusion
lpt = REFS.lthref + pxz;
pt = exp(lpt);
lp = REFS.lpref + rxz;
p = exp(lp);
P = REFS.pref;
PT = REFS.thref;
rho = p ./ (Rd * pt) .* (p0 * p.^(-1)).^(Rd / cp);
R = p ./ (Rd * PT) .* (p0 * P.^(-1)).^(Rd / cp);
RT = (rho .* pt) - (R .* PT);
UINF = norm(uxz - mean(mean(uxz)),Inf);
WINF = norm(wxz - mean(mean(wxz)),Inf);
RINF = norm(R - mean(mean(R)),Inf);
RTINF = norm(RT - mean(mean(RT)),Inf);
disp('Scaling constants for DynSGS coefficients:');
disp(['|| u - U_bar ||_max = ' num2str(UINF)]);
disp(['\| w - W_bar ||_max = ' num2str(WINF)]);
disp(['\| rho - rho_bar ||_max = ' num2str(RINF)]);
disp(['\| rhoTheta - rhoTheta_bar ||_max = ' num2str(RTINF)]);

%% Compute Ri, Convective Parameter, and BVF
DDZ_BC = REFS.DDZ;
dlrho = REFS.dlrref + REFS.sigma .* (DDZ_BC * (log(rho) - REFS.rref));
duj = REFS.dujref + REFS.sigma .* (DDZ_BC * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

DDZ_BC = REFS.DDZ;
dlpt = REFS.dlthref + REFS.sigma .* (DDZ_BC * real(pxz));
temp = p ./ (Rd * rho);
conv = temp .* dlpt;

RiREF = -BS.ga * REFS.dlrref(:,1);
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
ylabel('Elevation (km)');
title('Richardson Number');
xlim([0.1 1.0E4]);
ylim([0.0 30.0]);

subplot(1,2,2); plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks');
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
%
fig = figure('Position',[0 0 2000 1000]); fig.Color = 'w';
DDZ_BC = REFS.DDZ;
dlpres = REFS.dlpref + REFS.sigma .* (DDZ_BC * real(rxz));
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
%% Compute the nonlinearity parameter in Rho and plot...
%{
nlRho = exp(rxz) - 1.0;

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
plot(nlRho(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on;
xlabel('$\frac{\rho \prime}{\bar{\rho}}$','FontSize',30,'Interpreter','latex');
%ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
xlim([-1.0E-3 1.0E-3]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit([mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

fname = ['NLP_RHO_LnP' TestCase num2str(hC)];
drawnow;
screen2png(fname);

%% Compute the nonlinearity parameter in U and plot...
%
nlU = uxz ./ REFS.ujref;

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
plot(nlU(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on;
xlabel('$\frac{u \prime}{\bar{u}}$','FontSize',30,'Interpreter','latex');
%ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
xlim([-1.0E-1 1.0E-1]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit([mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

fname = ['NLP_U_LnP' TestCase num2str(hC)];
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
%
close all;
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER_LnP' char(TestCase) int2str(hC) '.mat'];
save(fileStore);
%}
