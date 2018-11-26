function [terrain,terrDeriv] = computeTopoDerivative(TestCase,xlon,DS,RAY)
    ht = 0.0 * xlon;
    dhdx = 0.0 * xlon;
    
    %% Make a windowing function and its derivative
    LI = 0.5 * (DS.L - 3 * RAY.width);
    wdex = find(abs(xlon) <= LI);    
    
    W = zeros(1, length(xlon));
    DW = zeros(1, length(xlon));
    [W(wdex), DW(wdex)] = tukeywin2(length(wdex), 0.5);
    
    %plot(xlon, W, xlon, DW); pause;

    %% Get the correct mountain for this test
    if ((strcmp(TestCase,'ShearJetSchar') == true) || ...
            (strcmp(TestCase,'ShearJetScharCBVF') == true) || ...
            (strcmp(TestCase,'ClassicalSchar') == true))
        ht = DS.hC * exp(-xlon.^2/DS.aC^2) .* (cos(pi * xlon / DS.lC)).^2;
       
        dhdx = -DS.hC * exp(-xlon.^2/DS.aC^2) .* ( ...
            2.0 * xlon / (DS.aC^2) .* (cos(pi * xlon / DS.lC)).^2 + ...
            pi/DS.lC * sin(2.0 * pi * xlon / DS.lC));
        
    elseif ((strcmp(TestCase,'HydroMtn') == true) || (strcmp(TestCase,'NonhydroMtn') == true))
        ht = DS.hC * (1.0 + xlon.^2 / DS.aC).^(-1);
        
        dhdx = -DS.hC * (2 * xlon / DS.aC) .* (1.0 + xlon.^2 / DS.aC).^(-2);
    elseif (strcmp(TestCase,'AndesMtn') == true)
        %AM = load('/Volumes/Patriot USB3/AndesTerrainDATA/EcuadorAndesProfiles400km.mat');
        AM = load('/home/jeguerra/Desktop/AndesTerrainDATA/EcuadorAndesProfiles400km.mat');
        xinp = AM.xlon;
        tpbkg = AM.(['tpavg' char(DS.hfilt)]);
        tpvar = AM.(['tpavg' char(DS.hfilt)]);
        tpful = tpbkg;% + tpvar;
        ht = zeros(size(tpvar,1),length(xlon));
        dhdx = ht;
        
        % Compute length scales and ratios for different domains
        Linp = 0.5 * (max(xinp) - min(xinp));
        L = 0.5 * (max(xlon) - min(xlon));
        dl = (L / Linp);
        scale = pi / L;
        
        % Compute the scaled interpolation grid [0 2pi]
        xhint = scale * (xlon - min(xlon));
        xhint = dl * xhint + pi * (1 - dl);
        for ii=1:size(tpful,1)
            %% Process the variance terrain data for height and slope
            % Interpolate the terrain height field
            [~,~,ht(ii,:)] = fourint(tpful(ii,:), xhint);
            ht(ii,:) = DS.hC * ht(ii,:);
            % Fourier terrain slopes at the native resolution
            dhdx30m = DS.hC * pi / Linp * fourdifft(tpful(ii,:),1);
            % Sample the terrain variance derivatives at the interpolated grid
            [~,~,dhdx(ii,:)] = fourint(dhdx30m, xhint);
            
            % DEBUG PLOTS WITH MULTIPLE COMPARISONS
            %{
            % First order slopes for comparison checks
            topo30m = DS.hC * tpful(ii,:);
            dhdx30m_O1 = topo30m;
            for jj=1:hsize
               if (jj == 1)
                   dhdx30m_O1(jj) = (topo30m(jj+1) - topo30m(jj)) / dx;
               elseif (jj == hsize)
                   dhdx30m_O1(jj) = (topo30m(jj) - topo30m(jj-1)) / dx;
               else
                   dhdx30m_O1(jj) = (topo30m(jj+1) - topo30m(jj-1)) / (2*dx);
               end
            end
            
            if ii == 3
                length(xinp)
                size(tpbkg)
                plot(xh, ht(ii,:), 's-', xinp, DS.hC * tpbkg(ii,:), 'o-');
                figure;
                plot(xinp, dhdx30m, 's-', xinp, dhdx30m_O1, 'o-', xh, dhdx(ii,:), '+-');
                grid on;
                %xlim([-1.0E4 1.0E4]);
                pause;
            end
            %}
        end
                
        % Pick one of the profiles... CHANGE THIS TO BE AUTOMATED
        ht = ht(3,:)';
        dhdx = dhdx(3,:)';
    end
    
    %terrain = W .* ht;
    %terrDeriv = W .* dhdx + DW .* ht;
    
    terrain = ht;
    terrDeriv = dhdx;
end