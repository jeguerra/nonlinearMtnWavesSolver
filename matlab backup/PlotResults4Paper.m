%% Plot some results

load 3072X451SpectralReferenceFFT1000.mat
%
%% Plot the solution frequency space
%
%SOL(sysDex) = sol;
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
mu = max(max(abs(uxz))); mu = 1.0;
mw = max(max(abs(wxz))); mw = 1.0;
mr = max(max(abs(rxz))); mr = 1.0;
mp = max(max(abs(pxz))); mp = 1.0;
xd = [1.0E-3 1.0E-1];
nc = 10;
%
figure;
%{
subplot(2,1,1);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / NX * uxz(:,kdex),nc,'LineStyle','-'); 
grid on;  colorbar;
caxis([-15.0 15.0]);
set(gca, 'XScale', 'log'); 
xlim(xd); ylim([0.0 40.0]);
title('Horizontal Velocity U $(ms^{-1})$');
ylabel('Height (km)');

subplot(2,1,2);
%}
colormap(cmap);
contourf((im2ikm * wfreq).^(-1), m2km * REFS.ZKL(:,kdex), ...
            1.0 / NX * wxz(:,kdex), nc, 'LineStyle','-'); 
grid on; colorbar;
caxis([-0.15 0.15]);
%set(gca, 'XScale', 'log'); 
%xlim([1.0E-2 2.0E-1]);
xlim([5.0 20.0]);
ylim([0.0 30.0]);
title('Vertical Velocity W $(ms^{-1})$');
xlabel('Wavelength (km)');
ylabel('Height (km)');
drawnow
fname = ['FREQ_RESP01' TestCase num2str(hC)];
drawnow;
export_fig(fname,'-png');
%{
figure;
subplot(2,1,1);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / NX * rxz(:,kdex),nc,'LineStyle','-'); grid on; %caxis([-1.0 1.0]);
set(gca, 'XScale', 'log'); xlim(xd);
title('$\ln \rho$');
ylabel('Height (km)');

subplot(2,1,2);
colormap(cmap);
contourf(im2ikm * wfreq,m2km * REFS.ZKL(:,kdex),1.0 / NX * pxz(:,kdex),nc,'LineStyle','-'); grid on; %caxis([-1.0 1.0]);
set(gca, 'XScale', 'log'); xlim(xd);
title('$\ln \theta$');
xlabel('Spatial Frequency $(km^{-1})$');
ylabel('Height (km)');
fname = ['FREQ_RESP02' TestCase num2str(hC)];
drawnow;
export_fig(fname,'-png');
%}
%% Compute Ri, Convective Parameter, and BVF
xdex = find(abs(REFS.XL(1,:)) <= 1.0E5);

close all; fig = figure;
fig.Position = [0 0 1800 1000];
subplot(1,2,1); semilogx(-Ri(:,xdex),1.0E-3*REFS.ZTL(:,xdex),'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
semilogx(RiREF,1.0E-3*REFS.ZTL(:,1),'r-s','LineWidth',1.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
ylabel('Height (km)');
title('Richardson Number');
xlim([0.0 1.0E5]);
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
export_fig(fname,'-png');

close all; figure;
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
drawnow;
export_fig(fname,'-png');

%%
close all; figure;
colormap(cmap);
contourf(XINT, ZINT, uxzint, 31); colorbar; grid on; cm = caxis;
%contourf(1.0E-3 * XI,1.0E-3 * ZI,ujref,31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XINT(1,:), 2.0E-3 * ZINT(1,:),'FaceColor','k'); hold off;
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
fname = ['UREferenceSolution' mtnh];
export_fig(fname,'-png');
%
close all; figure;
colormap(cmap);
contourf(XINT , ZINT, wxzint, 31); colorbar; grid on; cm = caxis;
hold on; area(1.0E-3 * XINT(1,:), 2.0E-3 * ZINT(1,:),'FaceColor','k'); hold off;
caxis(cm);
%xlim(1.0E-3 * [l1 + width l2 - width]);
%ylim(1.0E-3 * [0.0 zH - depth]);
%xlim([-100 100]);
%ylim([0 15]);
title('\textsf{$W^{\prime} ~~ (ms^{-1})$}');
xlabel('Distance (km)');
ylabel('Height (km)');
fname = ['WREferenceSolution' mtnh];
export_fig(fname,'-png');
