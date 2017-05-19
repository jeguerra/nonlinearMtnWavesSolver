function [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(prs, zH, z, Z)
    % Computes a background log pressure profile based on lapse rates and a
    % corresponding density background
    
    %% Initialize log pressure and log density
    lpref = zeros(size(Z));
    tref = lpref;
    dtref = lpref;

    %% Compute the troposphere layer log pressure and gradient
    tdex = find(z <= prs.HT);
    tref(tdex,:) = prs.T0 + prs.GAMT * Z(tdex,:);
    dtref(tdex,:) = prs.GAMT * ones(length(tdex), size(Z,2));
    scale = -prs.ga / (prs.Rd * prs.GAMT);
    lpref(tdex,:) =  scale * log(1.0 + (prs.GAMT / prs.T0) * Z(tdex,:)) + log(prs.p0);
    %{    
    dlpref(tdex,:) = -(prs.ga / prs.Rd) * tref(tdex,:).^(-1);
    
    % Compute the log density and density gradient in this layer
    rref = exp(lpref(tdex,:)) ./ (prs.Rd * tref(tdex,:));
    lrref(tdex,:) = log(rref);
    
    dlrref(tdex,:) = dlpref(tdex,:) - prs.GAMT * tref(tdex,:).^(-1);
    %}
    % Compute the temperature at the tropopause
    TTP = prs.T0 + prs.GAMT * prs.HT;
    % Compute the pressure at the tropopause
    PTP =  scale * log(1.0 + (prs.GAMT / prs.T0) * prs.HT) + log(prs.p0);

    %% Compute the mixed layer log pressure and gradient      
    mldex = find((z > prs.HT) & (z <= prs.HS));
    tref(mldex,:) = TTP * ones(length(mldex), size(Z,2));
    dtref(mldex,:) = zeros(length(mldex), size(Z,2));
    scale = -prs.ga / (prs.Rd * TTP);
    lpref(mldex,:) = scale * (Z(mldex,:) - prs.HT) + PTP;
    %{          
    dlpref(mldex,:) = -prs.ga / (prs.Rd * TTP);
               
    % Compute the log density and density gradient in this layer
    tref(mldex,:) = TTP * ones(length(mldex), size(Z,2));
    rref = exp(lpref(mldex,:)) ./ (prs.Rd * tref(mldex,:));
    lrref(mldex,:) = log(rref);
    
    dlrref(mldex,:) = dlpref(mldex,:);
    %}
    % Compute the temperature at the top of the mixed layer
    TML = TTP;
    % Compute the pressure at the top of the mixed layer
    PML = scale * (prs.HS - prs.HT) + PTP;

    %% Compute the stratosphere layer log pressure and gradient
    %sdex = find((z > prs.HS) & (z <= zH));
    sdex = find(z > prs.HS);
    tref(sdex,:) = TML + prs.GAMS * (Z(sdex,:) - prs.HS);
    dtref(sdex,:) = prs.GAMS * ones(length(sdex), size(Z,2));
    scale = -prs.ga / (prs.Rd * prs.GAMS);
    lpref(sdex,:) =  scale * log((TML + prs.GAMS * Z(sdex,:)) / (TML + prs.GAMS * prs.HS)) + PML;
    
    %% Compute the column gradient of log pressure
    dlpref = -(prs.ga / prs.Rd) * tref.^(-1);
             
    %% Compute the log density and density
    rref = exp(lpref) ./ (prs.Rd * tref);
    lrref = log(rref);
    
    dlrref = dlpref - dtref .* tref.^(-1);
    
    %plot(z, tref(:,1)); figure;
    %plot(z, dlrref(:,1)); 
    %pause;
    
    %% Compute the product of rho and P background
    %pref = exp(lpref);
    %rref = exp(lrref) .^ prs.GAM;
    

    %{
    % Compute the temperature at the stratopause
    TSP = TML + prs.GAMS * (zH - prs.HS);
    % Compute the pressure at the stratopause
    PSP =  prs.ga / (prs.Rd * prs.GAMS) * ...
                 log(1.0 - (prs.GAMS / TML) * (zH - prs.HS)) + PML;

    %% Compute the pressure drop for the rest of the atmosphere ISOTHERMAL
    rdex = find(z > zH);
    lpref(rdex,:) = prs.ga / (prs.Rd * TSP) * ...
                   (zH - Z(rdex,:)) + PSP;
               
    dlpref(rdex,:) = -prs.ga / (prs.Rd * TSP);
               
    % Compute the log density density gradient in this layer
    ltref = log(prs.Rd * TSP);
    lrref(rdex,:) = lpref(rdex,:) - ltref;
    
    dlrref(rdex,:) = dlpref(rdex,:);
    %}
end

