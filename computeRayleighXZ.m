function [rayField, BR] = computeRayleighXZ(prs,nu,depth,width,X,Z,applyTop,applyLateral)
    % Computes the static Rayleigh strength layers using cosine ramp
    dLayerZ = prs.zH - depth;
    dLayerR = prs.l2 - width;
    dLayerL = prs.l1 + width;
    
    % Get the indices for the lateral layers
    %[xr1,xr2] = ind2sub(size(X),find(X >= dLayerR));
    %[xl1,xl2] = ind2sub(size(X),find(X <= dLayerL));
    %xdex = [xldex xrdex];
    
    % Get the indices for the top layers
    %[zt1,zt2] = ind2sub(size(Z),find(Z >= dLayerZ));
    
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
    %% Assemble the layer strength field
    RLX = zeros(size(X));
    RLZ = zeros(size(Z));
    if applyLateral == true
        for ii=1:size(Z,1)
            for jj=1:size(Z,2)
                if X(ii,jj) >= dLayerR
                    XRL = X(ii,jj);
                    dNormX = (prs.l2 - XRL) / width;
                    RFX = 1.0 * nu * (0.0 + (cos(0.5 * pi * dNormX)).^4);
                elseif X(ii,jj) <= dLayerL
                    XRL = X(ii,jj);
                    dNormX = (XRL - prs.l1) / width;
                    RFX = 1.0 * nu * (0.0 + (cos(0.5 * pi * dNormX)).^4);
                end
                RLX(ii,jj) = RFX;
            end
        end
    end
    
    if applyTop == true
        for ii=1:size(Z,1)
            for jj=1:size(Z,2)
                if Z(ii,jj) >= dLayerZ
                    ZRL = Z(ii,jj); 
                    dNormZ = (prs.zH - ZRL) / depth;
                    RFZ = 1.0 * nu * (0.0 + (cos(0.5 * pi * dNormZ)).^4);

                    RLZ(ii,jj) = RFZ;
                end
            end
        end
    end
    
    % Get the overlapping indices
    RO = RLX .* RLZ;
    [oi,ok] = find(RO);
    
    RL = RLX + RLZ;
    
    % Handle the corner regions
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