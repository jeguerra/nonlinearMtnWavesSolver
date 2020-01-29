function [rayField, BR] = computeRayleighEllipse(prs,nu,depth,width,X,Z)
    % Computes the static Rayleigh strength layers using cosine ramp
    
    % Layer boundary curve parameters (4th order Ellipse)
    ae = prs.l2 - width;
    be = prs.zH - depth;
    
    % Compute Euclidean distance from the origin of all points
    DISP = sqrt(X.^2 + Z.^2);
    % Compute the slope of every ray emanating from the origin
    SLPR = Z ./ X;
    % Get the point where each ray intersects the layer boundary curve
    XE = ae * (1.0 + (ae * SLPR / be).^4).^(-0.25);
    
    ZE = be / ae * (ae^4 - XE.^4).^(0.25);
    ELP = sqrt(XE.^2 + ZE.^2);
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
                    ZI(ii,jj) = 0.0;
                elseif X(ii,jj) < 0.0
                    XI(ii,jj) = prs.l1;
                    ZI(ii,jj) = 0.0;
                end
            elseif SLPR(ii,jj) == Inf
                XI(ii,jj) = 0.0;
                ZI(ii,jj) = prs.zH;
            end
        end
    end
    
    % Compute Euclidean length of each ray to the boundary
    DISR = sqrt(XI.^2 + ZI.^2);
    
    %% Set up the layers around the ellipse boundary
    RL = 0.0 * DISR;
    for ii=1:size(DISR,1)
        for jj=1:size(DISR,2)
            if DISP(ii,jj) >= ELP(ii,jj)
                dNorm = (DISR(ii,jj) - DISP(ii,jj)) / (DISR(ii,jj) - ELP(ii,jj)); 
                RL(ii,jj) = nu * (cos(0.5 * pi * dNorm)).^4;
            end
        end
    end
    
    rayField = RL;
    %surf(X,Z,real(RL)); pause;
    
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