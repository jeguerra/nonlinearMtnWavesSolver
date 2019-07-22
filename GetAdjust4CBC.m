function [SOL,sysDex] = GetAdjust4CBC(REFS, BC, NX, NZ, OPS)
    %% Number of variables and block positions in the solution vector
    numVar = 4;

    %% Set an initial solution vector
    SOL = zeros(numVar*OPS,1);
    
    if BC == 0
        % Create boundary condition indices
        iW = 1;
        iT = 3;
        
        utdex = NZ:NZ:OPS;
        wtdex = utdex + iW*OPS;
        %ttdex = utdex + iT*OPS;    
        ubdex = 1:NZ:(OPS - NZ + 1);
        wbdex = ubdex + iW*OPS;
        %tbdex = ubdex + iT*OPS;
        
        disp('Hermite-Lagrange LogP-LogTheta Model...');
        %SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        %SOL(tbdex) = -REFS.ZTL(1,:) .* REFS.dlthref(1,:);
        
        %rowsOut = [wtdex ttdex wbdex tbdex];
        rowsOut = [wtdex wbdex];
           
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 1
        % Create boundary condition indices
        iW = 1;
        iT = 3;
        
        utdex = NZ:NZ:OPS;
        wtdex = utdex + iW*OPS;    
        ubdex = 1:NZ:(OPS - NZ + 1);
        wbdex = ubdex + iW*OPS;
        
        disp('Hermite-Lagrange Rho-RhoTheta Model, Dirichlet Lateral BC...');
        %SOL(wbdex) = REFS.DZT(1,:) .* REFS.rref(1,:) .* REFS.ujref(1,:);
        
        rowsOut = [wtdex wbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 2
        % Create boundary condition indices
        iW = 1;
        iT = 3;
        
        utdex = NZ:NZ:OPS;
        wtdex = utdex + iW*OPS;
        ptdex = utdex + iT*OPS;    
        ubdex = 1:NZ:(OPS - NZ + 1);
        wbdex = ubdex + iW*OPS;
        pbdex = ubdex + iT*OPS;
        
        disp('Applying BC FFT-Lagrange Model, Periodic Lateral BC');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        ZTLF = fft(REFS.ZTL, NX, 2);
        SOL(pbdex) = -ZTLF(1,:) .* REFS.dlthref(1,:);
        
        rowsOut = [utdex wtdex ptdex wbdex pbdex];
               
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    elseif BC == 3
        % Create boundary condition indices
        iW = 1;
        iP = 2;
        iT = 3;
        
        utdex = NZ:NZ:OPS;
        wtdex = utdex + iW*OPS;
        ptdex = utdex + iT*OPS;    
        ubdex = 1:NZ:(OPS - NZ + 1);
        wbdex = ubdex + iW*OPS;
        pbdex = ubdex + iT*OPS;
        
        % Get indices for the right boundary
        urdex = (OPS - NZ + 1):OPS;
        wrdex = urdex + iW*OPS;
        prdex = urdex + iP*OPS;
        trdex = urdex + iT*OPS;
        
        disp('Hermite-Lagrange LogP-LogTheta Model...');
        SOL(wbdex) = REFS.DZT(1,:) .* REFS.ujref(1,:);
        SOL(pbdex) = -REFS.ZTL(1,:) .* REFS.dlthref(1,:);
        
        rowsOut = [utdex wtdex ptdex wbdex pbdex];
        
        % Implement X periodicity
        %rightOut = [urdex wrdex prdex trdex];
        %rowsOut = [rowsOut rightOut];
           
        sysDex = setdiff(1:numVar*OPS, rowsOut);
    end
end
