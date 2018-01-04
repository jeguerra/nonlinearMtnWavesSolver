function [lpref,lrref,dlpref,dlrref] = computeBackgroundPressureCBVF(prs, Z)
    % Computes a background log pressure profile based on uniform
    % stratification.
    
    NBF = prs.BVF;
    
    %% Compute the potential temperature profile and gradient
    dlthref = NBF^2 / prs.ga;
    
    tref = prs.T0 * exp(NBF^2 / prs.ga * Z);
    epref = 1.0 + prs.ga^2 / (prs.cp * prs.T0 * NBF^2) * ...
        (exp(-NBF^2 / prs.ga * Z) - 1.0);
    
    rref = prs.p0 / prs.Rd * tref.^(-1) .* epref.^(prs.cv / prs.Rd);
    pref = prs.p0 * (prs.Rd / prs.p0 * rref .* tref).^(prs.cp / prs.cv);
    
    lpref = log(pref);
    lrref = log(rref);
    
    %% Compute the background log gradients... all constants in the CBVF case
    dlpref = -prs.ga * rref ./ pref;
    dlrref = 1.0 / prs.gam * dlpref - dlthref;
end

