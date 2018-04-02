clc
clear

H0010 = load('120X100SpectralReferenceHER_LnPShearJetSchar10.mat');
H0100 = load('120X100SpectralReferenceHER_LnPShearJetSchar100.mat');
H1000 = load('120X100SpectralReferenceHER_LnPShearJetSchar1000.mat');

%% Plot the 10m meter mountain
lpt = H0010.REFS.lthref + H0010.pxz;
pt = exp(lpt);
lp = H0010.REFS.lpref + H0010.rxz;
p = exp(lp);
rho = p ./ (H0010.Rd * pt) .* (H0010.p0 * p.^(-1)).^H0010.kappa;

DDZ_BC = H0010.REFS.DDZ;
dlrho = H0010.REFS.dlrref + H0010.REFS.sigma .* (DDZ_BC * (log(rho) - H0010.REFS.lrref));
duj = H0010.REFS.dujref + H0010.REFS.sigma .* (DDZ_BC * real(H0010.uxz));
Ri = -H0010.ga * dlrho ./ (duj.^2);

DDZ_BC = H0010.REFS.DDZ;
dlpt = H0010.REFS.dlthref + H0010.REFS.sigma .* (DDZ_BC * real(H0010.pxz));
temp = p ./ (H0010.Rd * rho);
conv = temp .* dlpt;

RiFig = figure;
subplot(1,3,1); semilogx(Ri,1.0E-3*H0010.REFS.ZTL,'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
ylabel('Elevation (km)');
title('10m Mountain');
xlim([0.1 1.0E4]);
ylim([0.0 30.0]);

ConvFig = figure;
subplot(1,3,1); plot(conv,1.0E-3*H0010.REFS.ZTL,'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * H0010.zH],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
ylabel('Elevation (km)');
title('10m Mountain');
ylim([0.0 30.0]);

%% Plot the 100m meter mountain
lpt = H0100.REFS.lthref + H0100.pxz;
pt = exp(lpt);
lp = H0100.REFS.lpref + H0100.rxz;
p = exp(lp);
rho = p ./ (H0100.Rd * pt) .* (H0100.p0 * p.^(-1)).^H0100.kappa;

DDZ_BC = H0100.REFS.DDZ;
dlrho = H0100.REFS.dlrref + H0100.REFS.sigma .* (DDZ_BC * (log(rho) - H0100.REFS.lrref));
duj = H0100.REFS.dujref + H0100.REFS.sigma .* (DDZ_BC * real(H0100.uxz));
Ri = -H0100.ga * dlrho ./ (duj.^2);

DDZ_BC = H0100.REFS.DDZ;
dlpt = H0100.REFS.dlthref + H0100.REFS.sigma .* (DDZ_BC * real(H0100.pxz));
temp = p ./ (H0100.Rd * rho);
conv = temp .* dlpt;

set(0,'CurrentFigure',RiFig);
subplot(1,3,2); semilogx(Ri,1.0E-3*H0100.REFS.ZTL,'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
%ylabel('Elevation (km)');
title('100m Mountain');
xlim([0.1 1.0E4]);
ylim([0.0 30.0]);

set(0,'CurrentFigure',ConvFig);
subplot(1,3,2); plot(conv,1.0E-3*H0100.REFS.ZTL,'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * H0100.zH],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
%ylabel('Elevation (km)');
title('100m Mountain');
ylim([0.0 30.0]);

%% Plot the 1000m meter mountain
lpt = H1000.REFS.lthref + H1000.pxz;
pt = exp(lpt);
lp = H1000.REFS.lpref + H1000.rxz;
p = exp(lp);
rho = p ./ (H1000.Rd * pt) .* (H1000.p0 * p.^(-1)).^H1000.kappa;

DDZ_BC = H1000.REFS.DDZ;
dlrho = H1000.REFS.dlrref + H1000.REFS.sigma .* (DDZ_BC * (log(rho) - H1000.REFS.lrref));
duj = H1000.REFS.dujref + H1000.REFS.sigma .* (DDZ_BC * real(H1000.uxz));
Ri = -H1000.ga * dlrho ./ (duj.^2);

DDZ_BC = H1000.REFS.DDZ;
dlpt = H1000.REFS.dlthref + H1000.REFS.sigma .* (DDZ_BC * real(H1000.pxz));
temp = p ./ (H1000.Rd * rho);
conv = temp .* dlpt;

set(0,'CurrentFigure',RiFig);
subplot(1,3,3); semilogx(Ri,1.0E-3*H1000.REFS.ZTL,'ks');
hold on;
semilogx([0.25 0.25],[0.0 1.0E5],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$Ri$');
%ylabel('Elevation (km)');
title('1000m Mountain');
xlim([0.1 1.0E4]);
ylim([0.0 30.0]);

set(0,'CurrentFigure',ConvFig);
subplot(1,3,3); plot(conv,1.0E-3*H1000.REFS.ZTL,'ks');
hold on;
semilogx([0.0 0.0],[0.0 1.0E-3 * H1000.zH],'k--','LineWidth',2.5);
hold off;
grid on; grid minor;
xlabel('$S_p$');
%ylabel('Elevation (km)');
title('1000m Mountain');
ylim([0.0 30.0]);

%% Save the two figures
figure(RiFig);
screen2png('RI_10-100-1000m');
%figure(ConvFig);
%screen2png('Sp_10-100-1000m');

