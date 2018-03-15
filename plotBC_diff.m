clc
clear
close all
startup

%% Load in the two sets of data
% Linear BC's on W and Vertical Momentum Equation Eliminated
BCJB = load('100X100_BielloBC_100m.mat');
% Linear BC's on W and LnP (continuity) Equation Eliminated
BCJG = load('100X100_GuerraBC_100m.mat');
% Coupled w' - h_x u' = h_x U and LnP (continuity) Equation Eliminated
%BCJG = load('100X80_GuerraInconsistentBC_100m.mat');

%% Stuff
% Use the NCL hotcold colormap
cmap = load('NCLcolormap254.txt');
cmap = cmap(:,1:3);

%% Plot the difference in the U and LnP fields where the discrepancy is most apparent (INTERPOLATED)
figure;
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * BCJB.XI,1.0E-3 * BCJB.ZI,BCJB.uxzint - BCJG.uxzint,31); colorbar; grid on;
title('$\Delta u^{\prime} ~ (ms^{-1})$');
subplot(1,2,2); contourf(1.0E-3 * BCJB.XI,1.0E-3 * BCJB.ZI,BCJB.rxzint - BCJG.rxzint,31); colorbar; grid on;
title('$\Delta (\ln p)^{\prime} ~ (Pa)$');
drawnow

%% Check the conserved entropy at the bottom boundary
entropyBiello = BCJB.pxz(1,:) + BCJB.REFS.ZTL(1,:) .* BCJB.REFS.dlthref(1,:);
entropyGuerra = BCJG.pxz(1,:) + BCJG.REFS.ZTL(1,:) .* BCJG.REFS.dlthref(1,:);
figure;
subplot(1,2,1); plot(1.0E-3 * BCJB.REFS.XL(1,:), entropyBiello,1.0E-3 * BCJB.REFS.XL(1,:), entropyGuerra, 'LineWidth', 1.5);
grid on;
title('BC Entropy');
legend('Biello BC','Guerra BC');
subplot(1,2,2); plot(1.0E-3 * BCJB.REFS.XL(1,:), entropyBiello - entropyGuerra, 'LineWidth', 1.5);
grid on;
title('$\Delta$ BC Entropy');

%% Check the local vertical derivative of u' (NOT INTERPOLATED, NATIVE GRID PLOTS)
figure;
colormap(cmap);
dUdZ_Biello = BCJB.DDZ_BC * BCJB.uxz;
dUdZ_Guerra = BCJG.DDZ_BC * BCJG.uxz;
colormap(cmap);
subplot(1,2,1); contourf(1.0E-3 * BCJB.REFS.XL,1.0E-3 * BCJB.REFS.ZTL,dUdZ_Biello,31); colorbar; grid on;
title('$(u^{\prime})_z$ Biello');
subplot(1,2,2); contourf(1.0E-3 * BCJB.REFS.XL,1.0E-3 * BCJB.REFS.ZTL,dUdZ_Guerra,31); colorbar; grid on;
title('$(u^{\prime})_z$ Guerra');
drawnow