%% COMPUTES STEADY LNP-LNT 2D MOUNTAIN WAVE PROBLEM IN 4 TEST CONFIGURATIONS:
%% SUPER HIGH RESOLUTION HORIZONTAL FFT SOLVER (PERIODIC BOUNDARIES ONLY)

% 1) 'ShearJetSchar' Discontinous background with strong shear
% 2) 'ShearJetScharCBVF' Uniform background with strong shear
% 3) 'ClassicalSchar' The typical Schar mountain test with uniform
% background and constant wind
% 4) 'AndesMtn' Same as 1) but with real input terrain data

clc
clear
close all
opengl info;
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 1024; % Expansion order matches physical grid
NZ = 360; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;
iW = 1;
iP = 2;
iT = 3;

%% Set the test case and global parameters
TestCase = 'ShearJetSchar'; BC = 2;
%TestCase = 'ShearJetScharCBVF'; BC = 2;
%TestCase = 'ClassicalSchar'; BC = 2;
%TestCase = 'AndesMtn'; BC = 2;

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
    hC = 1000.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '';
    u0 = 10.0;
    uj = 16.822;
    b = 1.386;
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    zH = 35000.0;
    l1 = - 1.0E4 * (3.0 * pi);
    l2 = 1.0E4 * (3.0 * pi);
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
    applyLateralRL = false;
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

figure;
contourf(m2km * REFS.XL, m2km * REFS.ZTL, REFS.ujref, 31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XINT(1,:), 2.0 * m2km * REFS.ZTL(1,:), 'FaceColor', 'k'); hold off;
caxis(cm);
xlim([l1 l2]);
ylim([0.0 zH]);
xlabel('Distance (km)');
ylabel('Height (km)');
title('Initial Flow Schematich - $U ms^{-1}$');
export_fig('UREF_INITIAL.png');
pause;

spparms('spumoni',2);
AN = LD(sysDex,sysDex);
b = (FF - LD * SOL); clear LD FF;
bN = b(sysDex,1);

%% Solve the coarse residual system by direct method
%solc = AN \ bN; clear AN;
solc = umfpack(AN, '\', bN); clear AN;
SOL(sysDex) = solc; clear solc;

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
%%
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
fname = ['FREQ_RESP01' TestCase num2str(hC) '.png'];
drawnow;
export_fig(fname);
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
fname = ['FREQ_RESP02' TestCase num2str(hC) '.png'];
drawnow;
export_fig(fname);
%pause;
%}
%% Plot the solution using the IFFT to recover the solution in physical space
uxz = real(ifft(reshape(SOL((1:OPS)),NZ,NX),[],2));
wxz = real(ifft(reshape(SOL((1:OPS) + OPS),NZ,NX),[],2));
rxz = real(ifft(reshape(SOL((1:OPS) + 2*OPS),NZ,NX),[],2));
pxz = real(ifft(reshape(SOL((1:OPS) + 3*OPS),NZ,NX),[],2));

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
dlpt = REFS.dlthref + REFS.sigma .* (DDZ_BC * real(pxz));
temp = p ./ (Rd * rho);
conv = temp .* dlpt;

duj = REFS.dujref + REFS.sigma .* (DDZ_BC * real(uxz));
Ri = ga * dlpt ./ (duj.^2);

RiREF = BS.ga * REFS.dlthref(:,1);
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

fname = ['RI_CONV_N2_' TestCase num2str(hC) '.png'];
drawnow;
export_fig(fname);
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

fname = ['FROUDE_' TestCase num2str(hC) '.png'];
drawnow;
export_fig(fname);
%}

%% INTERPOLATED GRID PLOTS
DNX = 256;
NXI = NX + DNX;
solint = ifft(reshape(SOL((1:OPS)), NZ, NX), [], 2);
uxz = real(interpft(solint, NXI, 2));
solint = ifft(reshape(SOL((1:OPS) + iW * OPS), NZ, NX), [], 2);
wxz = real(interpft(solint, NXI, 2));
solint = ifft(reshape(SOL((1:OPS) + iP * OPS), NZ, NX), [], 2);
rxz = real(interpft(solint, NXI, 2));
solint = ifft(reshape(SOL((1:OPS) + iT * OPS), NZ, NX), [], 2);
pxz = real(interpft(solint, NXI, 2));
NZI = 350;
[uxzint, XINT, ZINT, ~] = VertLegInterp(REFS, DS, RAY, uxz, NXI, NZI, 0);
[wxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, wxz, NXI, NZI, 0);
[rxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, rxz, NXI, NZI, 0);
[pxzint, ~, ~, ~] = VertLegInterp(REFS, DS, RAY, pxz, NXI, NZI, 0);

uxzint(:,end) = uxzint(:,1);
wxzint(:,end) = wxzint(:,1);
rxzint(:,end) = rxzint(:,1);
pxzint(:,end) = pxzint(:,1);

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
%% PLOTS
figure;
%colormap(cmap);
colormap('jet');
%contourf(XINT,ZINT,uxzint,20); colorbar; grid on; cm = caxis; 
contourf(1.0E-3 * XINT,1.0E-3 * ZINT, ujref, 21); colorbar; grid on; cm = caxis;
%hold on; area(1.0E-3 * XINT(1,:),2.0E-3 * ZINT(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-200 300]);
%ylim([0 15]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
%title('\textsf{$U^{\prime} ~~ (ms^{-1})$}');
title('\textsf{$\bar{U} ~~ (ms^{-1})$}');
%title('Initial Flow Schematic');
xlabel('Distance (km)');
ylabel('Height (km)');
export_fig(['UREferenceSolution' mtnh '.png']);
%%
figure;
colormap(cmap);
contourf(XINT,ZINT,wxzint,20); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XINT(1,:),1.0E-3 * ZINT(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Height (km)');
%screen2png(['WREferenceSolution' mtnh '.png']);
%%
figure;
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XINT,1.0E-3 * ZINT,rxzint,30); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln p)^{\prime} ~~ (Pa)$');
subplot(1,2,2); contourf(1.0E-3 * XINT,1.0E-3 * ZINT,pxzint,30); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('$(\ln \theta)^{\prime} ~~ (K)$');
drawnow

%%
%% Plot background fields including mean Ri number
zl = 1.0E-3 * REFS.ZTL(:,1);
figure;
suptitle('Background Profiles');
subplot(2,2,1);
plot(REFS.ujref(:,1),zl,'k-s','LineWidth',1.5); grid on;
xlabel('Speed $(m s^{-1})$');
ylabel('Height (km)');
ylim([0.0 35.0]);
drawnow;

subplot(2,2,2);
plot(REFS.pref(:,1) ./ (REFS.rref(:,1) * BS.Rd),zl,'k-s','LineWidth',1.5); grid on;
%title('Temperature Profile','FontSize',30);
xlabel('Temperature (K)');
ylabel('Height (km)');
ylim([0.0 35.0]);
drawnow;

subplot(2,2,3);
plot(REFS.thref(:,1),zl,'k-s','LineWidth',1.5); grid on;
xlabel('Potential Temperature (K)');
ylabel('Height (km)');
ylim([0.0 35.0]);
drawnow;

subplot(2,2,4);
plot(sqrt(BS.ga * REFS.dlthref(:,1)),zl,'k-s','LineWidth',1.5); grid on;
xlabel('Brunt-V\"ais\"ala Frequency ($s^{-1}$)');
ylabel('Height (km)');
ylim([0.0 35.0]);
drawnow;
export_fig('BACKGROUND_PROFILES.png');

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
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceFFT_' TestCase int2str(hC) '.mat'];
save(fileStore);
%}
