% Computes the semi-analytical solution to the steady, Euler equations
% in terrain following coordinates using a coordinate transformation from
% XZ to alpha-xi. The reference state is a standard atmosphere of piece-wise 
% linear temperature profiles with a smooth zonal jet given. Pressure and 
% density initialized in hydrostatic balance.

clc
clear
close all
%addpath(genpath('MATLAB/'))

%% Create the dimensional XZ grid
NX = 80; % Expansion order matches physical grid
NXO = 80; % Expansion order
NZ = 60; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;

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
if strcmp(TestCase,'ShearJetSchar') == true
    zH = 35000.0;
    l1 = -60000.0;
    l2 = 60000.0;
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
    zH = 30000.0;
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
    zH = 40000.0;
    l1 = -250000.0;
    l2 = 250000.0;
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
    width = 40000.0;
    nu1 = hfactor * 1.0E-2; nu2 = hfactor * 1.0E-2;
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-2;
    applyLateralRL = true;
    applyTopRL = true;
    aC = 5000.0;
    lC = 4000.0;
    hC = 100.0;
    mtnh = [int2str(hC) 'm'];
    hfilt = '25km';
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

%% Compute the grid and initialization state
REFS = computeGridInitializationNL(DS, BS, UJ, RAY, TestCase, NXO, NX, NZ, applyTopRL, applyLateralRL);

SOL = zeros(4 * OPS, 1);
ruxz = SOL(1:OPS);
rwxz = SOL((1:OPS) + OPS);
rxz = SOL((1:OPS) + 2*OPS);
pxz = SOL((1:OPS) + 3*OPS);

iter = 2;
for n=1:iter
    %% Compute the LHS of the Newton update
    [LD, FF] = evaluateJacobianOperatorCBC_FluxForm(ruxz, rwxz, rxz, pxz, BS, REFS, RAY, n);

    %% Get the boundary conditions
    sysDex = computeBCIndexNL(BC,NX,NZ,OPS);
    
    if n == 1
        disp('Solve the raw system with matlab default \.');
        tic
        spparms('spumoni',2);
        A = LD(sysDex,sysDex);
        b = FF(sysDex,1);
        sol = A \ b;
        clear A b;
        toc
        
        %% Get the solution fields
        SOL(sysDex) = sol;
        clear sol;
        %
        ruxz = reshape(ruxz + SOL((1:OPS)),NZ,NX);
        rwxz = reshape(rwxz + SOL((1:OPS) + OPS),NZ,NX);
        rxz = reshape(rxz + SOL((1:OPS) + 2*OPS),NZ,NX);
        pxz = reshape(pxz + SOL((1:OPS) + 3*OPS),NZ,NX);

        ruxz(:,end) = ruxz(:,1);
        rwxz(:,end) = rwxz(:,1);
        rxz(:,end) = rxz(:,1);
        pxz(:,end) = pxz(:,1);

        %% Send the current iterate back...
        ruxz = reshape(ruxz, OPS, 1);
        rwxz = reshape(rwxz, OPS, 1);
        rxz = reshape(rxz, OPS, 1);
        pxz = reshape(pxz, OPS, 1);
    elseif n == 2
        S = svds(LD, 5);
        LB = 1.0 / max(S);
        
        qdex = 1:OPS;
        disp(['Lower Error Bound in RhoU: ' num2str(LB * norm(FF(qdex)))]);
        disp(['Lower Error Bound in RhoW: ' num2str(LB * norm(FF(qdex + OPS)))]);
        disp(['Lower Error Bound in Rho: ' num2str(LB * norm(FF(qdex + 2*OPS)))]);
        disp(['Lower Error Bound in RhoTheta: ' num2str(LB * norm(FF(qdex + 3*OPS)))]);
        
        disp(['Non-linear Residual in RhoU: ' num2str(norm(FF(qdex)))]);
        disp(['Non-linear Residual in RhoW: ' num2str(norm(FF(qdex + OPS)))]);
        disp(['Non-linear Residual in Rho: ' num2str(norm(FF(qdex + 2*OPS)))]);
        disp(['Non-linear Residual in RhoTheta: ' num2str(norm(FF(qdex + 3*OPS)))]);
        
        evaluateResidual = true;
        if evaluateResidual
            ruxz = reshape(FF((1:OPS)),NZ,NX);
            rwxz = reshape(FF((1:OPS) + OPS),NZ,NX);
            rxz = reshape(FF((1:OPS) + 2*OPS),NZ,NX);
            pxz = reshape(FF((1:OPS) + 3*OPS),NZ,NX);
        end
        
        ruxz = reshape(ruxz,NZ,NX);
        rwxz = reshape(rwxz,NZ,NX);
        rxz = reshape(rxz,NZ,NX);
        pxz = reshape(pxz,NZ,NX);

        ruxz(:,end) = ruxz(:,1);
        rwxz(:,end) = rwxz(:,1);
        rxz(:,end) = rxz(:,1);
        pxz(:,end) = pxz(:,1);
    end
end

%% Check the residuals of the first iteration (linear reference)
%{
ruxz = reshape(FF((1:OPS)),NZ,NX);
rwxz = reshape(FF((1:OPS) + OPS),NZ,NX);
rxz = reshape(FF((1:OPS) + 2*OPS),NZ,NX);
pxz = reshape(FF((1:OPS) + 3*OPS),NZ,NX);
%}

%% Compute the kinematic fields (u, w, rho, rho-theta)
uxz = (REFS.rref .* REFS.ujref + ruxz) ./ (REFS.rref + rxz) - REFS.ujref;
wxz = rwxz ./ (REFS.rref + rxz);

%% Interpolate to a regular grid using Hermite and Legendre transforms'
%
NXI = 600;
NZI = 300;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, real(pxz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
%% Interpolate to a regular grid using Hermite and Laguerre transforms'
%{
DXI = 100.0;
DZI = 50.0;
xint = l1:DXI:l2; NXI = length(xint);
zint = 0.0:DZI:zH; NZI = length(zint);
%[zint, ~] = lagdifJEG(NZI, 1, 2.0 * DS.zH);
[uxzint, XINT, ZINT, ZLINT] = HerTransLagTrans(REFS, DS, real(uxz), NXI, NZI, xint, zint);
[wxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(wxz), NXI, NZI, xint, zint);
[rxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(rxz), NXI, NZI, xint, zint);
[pxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(pxz), NXI, NZI, xint, zint);

XI = l2 * XINT;
ZI = ZINT;
%}
%% Plot the solution in the native grids
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
title('Perturbation Density (kg/m^3)');
subplot(1,2,2); contourf(REFS.XL,REFS.ZTL,real(pxz),31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Perturbation Theta (K)');
drawnow
%}

% Use the NCL hotcold colormap
cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

% INTERPOLATED GRID PLOTS
%
width = 0.0;
depth = 0.0;
%% Compute the reference state initialization
%
if strcmp(TestCase,'ShearJetSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ShearJetScharCBVF') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [lprefU,~,dlprefU,~] = computeBackgroundPressureCBVF(BS, ZLINT);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
elseif strcmp(TestCase,'ClassicalSchar') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressureCBVF(BS, ZINT);
    [ujref, ~] = computeJetProfileUniform(UJ, lpref);
elseif strcmp(TestCase,'AndesMtn') == true
    [lpref, lrref, dlpref, dlrref] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZINT, RAY);
    [lprefU,~,dlprefU,~] = computeBackgroundPressure(BS, zH, ZINT(:,1), ZLINT, RAY);
    [ujref,dujref] = computeJetProfile(UJ, BS.p0, lprefU, dlprefU);
end
%
dlthref = 1.0 / BS.gam * dlpref - dlrref;

%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
contourf(XI,ZI,ujref,31); colorbar;
xlim([l1 l2]);
ylim([0.0 zH]);
title('Background Horizontal Velocity U (m/s)');
%}
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref + uxzint,31); colorbar; grid on;
%hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('Total Horizontal Velocity U $(ms^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['UREferenceSolution' mtnh '.png']);
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on;
%hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
%caxis([-0.08 0.08]);
title('Vertical Velocity W $(ms^{-1})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['WREferenceSolution_FluxForm' mtnh '.png']);
%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI, ... 
    (exp(lrref + rxzint) .* (ujref + uxzint) .* wxzint),31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
title('Momentum Flux $(kg~m^{-1}~s^{-2})$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['MFluxREferenceSolution' mtnh '.png']);
drawnow
%}
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,rxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Perturbation Density $(kg m^{-3})$','FontWeight','normal','Interpreter','latex');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,pxzint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Perturbation $\rho \theta$ $(K kg m^{-3})$','FontWeight','normal','Interpreter','latex');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,abs(uxzint) ./ abs(ujref),31,'LineStyle','none'); colorbar;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
title('Nonlinearity Ratio $\left| \frac{u}{\bar{u}} \right|$','FontSize',32,'FontWeight','normal','Interpreter','latex');
%}
drawnow
%
%% Compute the local Ri number and plot ...
%
dlrho = REFS.dlrref + REFS.sigma .* (REFS.DDZ_L * real(rxz));
duj = REFS.dujref + REFS.sigma .* (REFS.DDZ_L * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

RiREF = -BS.ga * REFS.dlrref(:,1);
RiREF = RiREF ./ (REFS.dujref(:,1).^2);

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
semilogx(Ri(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',1.5);
semilogx(RiREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on;
xlabel('Ri','FontSize',30);
%ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
xlim([0.1 1.0E4]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit([mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'RI_TEST_' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%{
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
[dujint, ~, ~] = HerTransLegInterp(REFS, DS, real(duj), NXI, NZI, 0, 0);
contourf(1.0E-3 * XI,1.0E-3 * ZI,dujint,31); colorbar; grid on;
xlim(1.0E-3 * [l1 + width l2 - width]);
ylim(1.0E-3 * [0.0 zH - depth]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
title('Vertical Shear $s^{-1}$','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['VerticalShear' mtnh '.png']);
drawnow
%}

%% Compute the local convective stability parameter and plot...
%
theta = REFS.thref + real(pxz);
dpt = REFS.thref .* REFS.dlthref + (REFS.DDZ_L * real(pxz));
dlpt = dpt ./ theta;
conv = dlpt;

xdex = 1:1:NX;
fig = figure('Position',[0 0 800 1200]); fig.Color = 'w';
plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on;
xlabel('$\frac{d \ln \theta}{d z}$','FontSize',30,'Interpreter','latex');
%ylabel('Altitude (km)','FontSize',30);
ylim([0.0 1.0E-3*zH]);
%xlim([0.1 1.0E3]);
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

pt = mtit([mtnh ' Mountain'],'FontSize',36,'FontWeight','normal','Interpreter','tex');

dirname = '../ShearJetSchar/';
fname = [dirname 'CONV_TEST_' TestCase num2str(hC)];
drawnow;
screen2png(fname);
%}

%% Compute the nonlinearity parameter in Rho and plot...
%
nlRho = real(rxz) ./ (REFS.rref);

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

dirname = '../ShearJetSchar/';
fname = [dirname 'NLP_RHO_FluxForm' TestCase num2str(hC)];
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

dirname = '../ShearJetSchar/';
fname = [dirname 'NLP_U_FluxForm' TestCase num2str(hC)];
drawnow;
screen2png(fname);

%% Debug
%
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(2,2,1); surf(XI,ZI,uxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Total Horizontal Velocity U (m/s)');
subplot(2,2,2); surf(XI,ZI,wxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Vertical Velocity W (m/s)');
subplot(2,2,3); surf(XI,ZI,rxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Perturbation Density (kg/m^3)');
subplot(2,2,4); surf(XI,ZI,pxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Perturbation Pressure (Pa)');
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
fileStore = [int2str(NX) 'X' int2str(NZ) 'SpectralReferenceHER_FluxForm' char(TestCase) int2str(hC) '.mat'];
save(fileStore);
%}
