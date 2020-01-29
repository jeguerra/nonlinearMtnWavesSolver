function [rayField, BR] = computeRayleighPolar(prs,nu,depth,X,Z)
    % Computes the static Rayleigh strength layers using cosine ramp
    
    % Compute Euclidean distance from the origin of all points
    DISP = sqrt(X.^2 + Z.^2);
    % Compute the slope of every ray emanating from the origin
    SLPR = Z ./ X;
    % Get the point where each ray intersects the boundary
    XI = 0.0 * X;
    ZI = 0.0 * Z;
    for ii=1:size(SLPR,1)
        for jj=1:size(SLPR,2)
            % Do the top right quadrant
            if (SLPR(ii,jj) > 0.0 && SLPR(ii,jj) ~= Inf)
                XI(ii,jj) = prs.zH / SLPR(ii,jj);
                if XI(ii,jj) > prs.l2
                    XI(ii,jj) = prs.l2;
                    ZI(ii,jj) = SLPR(ii,jj) * XI(ii,jj);
                else
                    ZI(ii,jj) = prs.zH;
                end
            end
            % Do the top left quadrant
            if (SLPR(ii,jj) < 0.0 && SLPR(ii,jj) ~= Inf)
                XI(ii,jj) = prs.zH / SLPR(ii,jj);
                if XI(ii,jj) < prs.l1
                    XI(ii,jj) = prs.l1;
                    ZI(ii,jj) = SLPR(ii,jj) * XI(ii,jj);
                else
                    ZI(ii,jj) = prs.zH;
                end
            end
            % Handle horizontal and vertical points
            if SLPR(ii,jj) == 0.0
                if X(ii,jj) > 0.0
                    XI(ii,jj) = prs.l2;
                elseif X(ii,jj) < 0.0
                    XI(ii,jj) = prs.l1;
                end
            elseif SLPR(ii,jj) == Inf
                ZI(ii,jj) = prs.zH;
            end
        end
    end
    
    % Compute Euclidean length of each ray to the boundary
    DISR = sqrt(XI.^2 + ZI.^2);
    
    % Compute the ratio of distance of a point to the boundary along rays
    DR = DISP ./ DISR;
    
    %% Set up the layers assuming a fraction of the overal dimension
    dN = depth / prs.zH;
    
    RL = 0.0 * DR;
    for ii=1:size(DR,1)
        for jj=1:size(DR,2)
            if DR(ii,jj) >= (1.0 - dN)
                dNorm = (1.0 - DR(ii,jj)) / dN;
                RL(ii,jj) = 0.5 * nu * (1.0 + cos(pi * dNorm));
            end
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