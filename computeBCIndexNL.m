function sysDex = computeBCIndexNL(BC,NX,NZ,OPS)
    numVar = 4;

    %% Create boundary condition indices
    uldex = 2:NZ-1;
    wldex = uldex + OPS;
    rldex = uldex + 2*OPS;
    pldex = uldex + 3*OPS;
    LeftOutExcludeCorners = [uldex wldex rldex pldex];
    RightOutExcludeCorners = LeftOutExcludeCorners + (OPS - NZ);
    
    if BC == 0
        disp('Hermite-Lagrange LogP-LogTheta Model, Dirichlet Lateral BC...');
        % LnP and Theta at the corners
        LeftCorners = [(2*OPS + 1) (3*OPS + 1) ...
                       (2*OPS + 1) (3*OPS + 1)];
        RightCorners = [(3*OPS - NZ + 1) (4*OPS - NZ + 1) ...
                        3*OPS 4*OPS];
        rowsOut = [LeftOutExcludeCorners LeftCorners ...
                   RightOutExcludeCorners RightCorners];
    elseif BC == 1
        disp('Hermite-Lagrange Rho-RhoTheta Model, Dirichlet Lateral BC...');
        % LnP and Theta at the corners
        LeftCorners = [(2*OPS + 1) ...
                       (2*OPS + 1)];
        RightCorners = [(3*OPS - NZ + 1) ...
                        3*OPS];
        rowsOut = [LeftOutExcludeCorners LeftCorners ...
                   RightOutExcludeCorners RightCorners];
    elseif BC == 2
        disp('Applying BC FFT-Lagrange Model...');
        rowsOut = [];
    end


    sysDex = setdiff(1:numVar*OPS + 2*NX, rowsOut);
end