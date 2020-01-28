function [lpref,lrref,dlpref,dlrref] = computeBackgroundPressure(prs, zH, z, Z, RAY)
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
    % Compute the temperature at the top of the mixed layer
    TML = TTP;
    % Compute the pressure at the top of the mixed layer
    PML = scale * (prs.HS - prs.HT) + PTP;

    %% Compute the stratosphere layer log pressure and gradient
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
    
    %% Fix the background fields in the Rayleigh layer (null out wave growth)
    %{
    rdex = find(z >= (zH - RAY.depth));
    
    rq = tref(rdex(1),:);
    tref(rdex,:) = repmat(rq,length(rdex),1);
    
    rq = lrref(rdex(1),:);
    lrref(rdex,:) = repmat(rq,length(rdex),1);
    
    rq = lpref(rdex(1),:);
    lpref(rdex,:) = repmat(rq,length(rdex),1);
    
    dlrref(rdex,:) = 0.0 * dlrref(rdex,:);
    dlpref(rdex,:) = 0.0 * dlpref(rdex,:);
    %}
    %plot(z, tref(:,1)); figure;
    %plot(z, dlrref(:,1)); 
    %pause;
end

