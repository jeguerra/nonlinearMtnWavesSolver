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
NX = 100; % Expansion order matches physical grid
NXO = 80; % Expansion order
NZ = 80; % Expansion order matches physical grid
OPS = NX * NZ;
numVar = 4;

%% Set the test case and global parameters
%TestCase = 'ShearJetSchar'; BC = 1;
TestCase = 'ShearJetScharCBVF'; BC = 0;
%TestCase = 'ClassicalSchar'; BC = 0;
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
    l1 = - 1.0E4 * (2.0 * pi);
    l2 = 1.0E4 * (2.0 * pi);
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
    nu3 = hfactor * 1.0E-2; nu4 = hfactor * 1.0E-6;
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
    hC = 0.1;
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
REFS = computeGridInitializationNL(DS, BS, UJ, RAY, TestCase, NX, NZ, applyTopRL, applyLateralRL);

SOL = zeros(4 * OPS, 1);
ruxz = SOL(1:OPS);
rwxz = SOL((1:OPS) + OPS);
rxz = SOL((1:OPS) + 2*OPS);
pxz = SOL((1:OPS) + 3*OPS);

iter = 2;
for n=1:iter
    %% Compute the LHS of the Newton update
    [LD, FF, RR, UREF, RREF, RTHREF] = evaluateJacobianOperatorCBC_FluxForm(ruxz, rwxz, rxz, pxz, BS, REFS, RAY);

    %% Solve the system for iteration 1 only
    if n == 1
        sysDex = computeBCIndexNL(BC,NX,NZ,OPS);
        disp('Solve the raw system with matlab default \.');
        tic
        spparms('spumoni',2);
        A = LD(sysDex,sysDex);
        b = FF(sysDex,1);
        clear LD FF;
        AN = A' * A;
        bN = A' * b;
        clear A b;
        sol = AN \ bN;
        clear AN bN;
        toc
        %% Get the solution fields
        SOL(sysDex) = sol;
        clear sol;
    else
        return;
    end
    
    %% Send the current iterate back
    ruxz = ruxz + SOL(1:OPS);
    rwxz = rwxz + SOL((1:OPS) + OPS);
    rxz = rxz + SOL((1:OPS) + 2*OPS);
    pxz = pxz + SOL((1:OPS) + 3*OPS);
end

evaluateResidual = false;
if evaluateResidual
    ruxz = reshape(RR((1:OPS)),NZ,NX);
    rwxz = reshape(RR((1:OPS) + OPS),NZ,NX);
    rxz = reshape(RR((1:OPS) + 2*OPS),NZ,NX);
    pxz = reshape(RR((1:OPS) + 3*OPS),NZ,NX);
else
    ruxz = reshape(ruxz,NZ,NX);
    rwxz = reshape(rwxz,NZ,NX);
    rxz = reshape(rxz,NZ,NX);
    pxz = reshape(pxz,NZ,NX);
end

%% Compute the kinematic fields from (u, w, rho, rho-theta)
uxz = (REFS.rref .* REFS.ujref + ruxz) ./ (REFS.rref + rxz) - REFS.ujref;
wxz = rwxz ./ (REFS.rref + rxz);
txz = (REFS.rref .* REFS.thref + pxz) ./ (REFS.rref + rxz) - REFS.thref;

%% Interpolate to a regular grid using Hermite and Legendre transforms'
%
NXI = 2000;
NZI = 300;
[uxzint, XINT, ZINT, ZLINT] = HerTransLegInterp(REFS, DS, RAY, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLegInterp(REFS, DS, RAY, real(txz), NXI, NZI, 0, 0);

XI = l2 * XINT;
ZI = ZINT;
%}
%% Interpolate to a regular grid using Hermite and Laguerre transforms'
%{
NXI = 600;
NZI = 300;
[uxzint, XINT, ZINT, ZLINT] = HerTransLagTrans(REFS, DS, real(uxz), NXI, NZI, 0, 0);
[wxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(wxz), NXI, NZI, 0, 0);
[rxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(rxz), NXI, NZI, 0, 0);
[pxzint, ~, ~] = HerTransLagTrans(REFS, DS, real(pxz), NXI, NZI, 0, 0);

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
    %[ujref,dujref] = computeJetProfile(UJ, BS.p0, lpref, dlpref);
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
%}
dlthref = 1.0 / BS.gam * dlpref - dlrref;

fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,uxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
disp(['U MAX: ' num2str(max(max(uxz)))]);
disp(['U MIN: ' num2str(min(min(uxz)))]);
title('\textsf{Total Horizontal Velocity U $(m~s^{-1})$}','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
fig.CurrentAxes.FontSize = 30; fig.CurrentAxes.LineWidth = 1.5;
screen2png(['UREferenceSolution' mtnh '.png']);
%
fig = figure('Position',[0 0 1800 1200]); fig.Color = 'w';
colormap(cmap);
contourf(1.0E-3 * XI,1.0E-3 * ZI,wxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
title('\textsf{Vertical Velocity W $(m~s^{-1})$}','FontWeight','normal','Interpreter','latex');
xlabel('Distance (km)');
ylabel('Elevation (km)');
screen2png(['WREferenceSolution' mtnh '.png']);
%
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * XI,1.0E-3 * ZI,rxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
title('Perturbation Density $(kg m^{-3})$','FontWeight','normal','Interpreter','latex');
subplot(1,2,2); contourf(1.0E-3 * XI,1.0E-3 * ZI,pxzint,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XI(1,:),1.0E-3 * ZI(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
title('Perturbation Potential Temperature (K)','FontWeight','normal','Interpreter','latex');
drawnow

%% Compute the scaling constants needed for residual diffusion
rho = REFS.rref + rxz;
theta = REFS.thref + txz;
PT = REFS.thref;
RT = REFS.rref .* REFS.thref;
RhoTheta = RT + pxz;
UINF = norm(uxz - mean(mean(uxz)),Inf);
WINF = norm(wxz - mean(mean(wxz)),Inf);
R = exp(rxz);
RINF = norm(R - mean(mean(rxz)),Inf);
RTINF = norm(RT - mean(mean(pxz)),Inf);
disp('Scaling constants for DynSGS coefficients:');
disp(['|| u - U_bar ||_max = ' num2str(UINF)]);
disp(['\| w - W_bar ||_max = ' num2str(WINF)]);
disp(['\| rho - rho_bar ||_max = ' num2str(RINF)]);
disp(['\| rhoTheta - rhoTheta_bar ||_max = ' num2str(RTINF)]);

%% Compute Ri, Convective Parameter, and BVF
%{
DDZ_BC = REFS.DDZ;
dlrho = rho.^(-1) .* (REFS.drref + REFS.sigma .* (DDZ_BC * real(rxz)));
duj = REFS.dujref + REFS.sigma .* (DDZ_BC * real(uxz));
Ri = -ga * dlrho ./ (duj.^2);

DDZ_BC = REFS.DDZ;
dlpt = theta.^(-1) .* (REFS.dthref + REFS.sigma .* (DDZ_BC * real(txz)));
temp = ((Rd / p0)^kappa * rho.^kappa .* (PT + txz)).^(1.0 / (1.0 - kappa));
conv = temp .* dlpt;

RiREF = -BS.ga * REFS.dlrref(:,1);
RiREF = RiREF ./ (REFS.dujref(:,1).^2);

xdex = 1:1:NX;
fig = figure('Position',[0 0 2000 1000]); fig.Color = 'w';
subplot(1,3,1); semilogx(Ri(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',1.5);
semilogx(RiREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
xlabel('Ri');
ylabel('Elevation (km)');
title('Ri Number','FontWeight','normal','Interpreter','latex');
xlim([0.1 1.0E4]);
ylim([0.0 1.0E-3 * zH]);
fig.CurrentAxes.FontSize = 24; fig.CurrentAxes.LineWidth = 1.5;

subplot(1,3,2); plot(conv(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5);
grid on; grid minor;
xlabel('\textsf{$\frac{T}{\theta} \frac{d \theta}{d z}$}','Interpreter','latex');
%ylabel('Elevation (km)');
title('Convective Stability','FontWeight','normal','Interpreter','latex');
%ylim([0.0 30.0]);
%xlim([0.1 1.0E3]);
fig.CurrentAxes.FontSize = 24; fig.CurrentAxes.LineWidth = 1.5;

NBVF = (ga .* dlpt);

NBVF_REF = (ga * (1 / gam * REFS.dlpref - REFS.dlrref));

xdex = 1:1:NX;
subplot(1,3,3); plot(NBVF(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks','LineWidth',1.5); hold on;
plot(NBVF_REF(:,1),1.0E-3*REFS.ZTL(:,1),'ro-','LineWidth',1.5); hold off;
grid on; grid minor;
title('Brunt-V\"ais\"al\"a','FontWeight','normal','Interpreter','latex');
xlabel('\textsf{$\mathcal{N}^2$}','Interpreter','latex');
%ylabel('\textsf{Altitude (km)}','Interpreter','latex');
%ylim([0.0 30.0]);
%xlim([0.1 1.0E3]);
fig.CurrentAxes.FontSize = 24; fig.CurrentAxes.LineWidth = 1.5;
drawnow;

fname = ['RI_CONV_N2_' TestCase num2str(hC)];
drawnow;
screen2png(fname);

%% Compute the nonlinearity parameter in Rho and plot...
%
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
fig = figure('Position',[0 0 1600 1200]); fig.Color = 'w';
subplot(2,2,1); surf(XI,ZI,uxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Total Horizontal Velocity U (m/s)');
subplot(2,2,2); surf(XI,ZI,wxzint); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Vertical Velocity W (m/s)');
subplot(2,2,3); surf(XI,ZI,exp(lrref) .* (exp(rxzint) - 1.0)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
title('Perturbation Density (kg/m^3)');
subplot(2,2,4); surf(XI,ZI,exp(lpref) .* (exp(pxzint) - 1.0)); colorbar; xlim([-10000.0 30000.0]); ylim([0.0 5000.0]);
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

