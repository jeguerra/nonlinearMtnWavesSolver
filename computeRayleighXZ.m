function [rayField, BR] = computeRayleighXZ(prs,nu,depth,width,X,Z,applyTop,applyLateral)
    % Computes the static Rayleigh strength layers using cosine ramp
    dLayerZ = prs.zH - depth;
    dLayerR = prs.l2 - width;
    dLayerL = prs.l1 + width;
    
    % Get the indices for the lateral layers
    [xr1,xr2] = ind2sub(size(X),find(X > dLayerR));
    [xl1,xl2] = ind2sub(size(X),find(X < dLayerL));
    %xdex = [xldex xrdex];
    
    % Get the indices for the top layers
    [zt1,zt2] = ind2sub(size(Z),find(Z > dLayerZ));
    
    % Get the layer physical locations
    XRL1 = X(xl1,xl2);
    XRL2 = X(xr1,xr2);
    ZRL = Z(zt1,zt2);
    RLX = zeros(size(X));
    clear X Z;
    %% Set up the layers (default cosine profiles)
    %{
    dNormZ = (prs.zH - ZRL) / depth;
    RFZ = 0.5 * nu * (1.0 + cos(pi * dNormZ));
    
    dNormX = (prs.l2 - XRL2) / width;
    RFX2 = 0.5 * nu * (1.0 + cos(pi * dNormX));
    
    dNormX = (XRL1 - prs.l1) / width;
    RFX1 = 0.5 * nu * (1.0 + cos(pi * dNormX));
    %}
    %% 2nd or 4th order profiles
    %
    dNormZ = (prs.zH - ZRL) / depth;
    RFZ = 1.0 * nu * (0.0 + (cos(0.5 * pi * dNormZ)).^4);
    clear dNormZ;
    
    dNormX = (prs.l2 - XRL2) / width;
    RFX2 = 1.0 * nu * (0.0 + (cos(0.5 * pi * dNormX)).^4);
    clear dNormX;
    
    dNormX = (XRL1 - prs.l1) / width;
    RFX1 = 1.0 * nu * (0.0 + (cos(0.5 * pi * dNormX)).^4);
    clear dNormX;
    %}
    
    %% Assemble the layer strength field
    if applyLateral == true
        RLX(xl1,xl2) = RFX1;
        RLX(xr1,xr2) = RFX2;
    end
    
    RLZ = 0.0 * RLX;
    if applyTop == true
        RLZ(zt1,zt2) = RFZ;
    end
    
    % Get the overlaping indices
    RO = RLX .* RLZ;
    [oi,ok] = find(RO);
    
    RL = RLX + RLZ;
    
    % Average the overlaping corners
    %RL(oi,ok) = 0.5 * (RLX(oi,ok) + RLZ(oi,ok));
    %RL(oi,ok) = sqrt(RLX(oi,ok).^2 + RLZ(oi,ok).^2);
    for ii = 1:length(oi)
        for kk = 1:length(ok)
            RL(oi(ii),ok(kk)) = ...
                max([RLX(oi(ii),ok(kk)) RLZ(oi(ii),ok(kk))]);
        end
    end
    
    rayField = RL;
    %surf(X,Z,RL); pause;
    
    %% Get the binary matrix for layers only
    BR = rayField;
    SBR = size(BR);
    for ii=1:SBR(1)
        for jj=1:SBR(2)
            if BR(ii,jj) ~= 0.0
                BR(ii,jj) = rayField(ii,jj) / rayField(ii,jj);
            end
        end
    end
end