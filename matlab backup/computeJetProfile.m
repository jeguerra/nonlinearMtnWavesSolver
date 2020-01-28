function [ujref,dujref] = computeJetProfile(prs, p0, lpref, dlpref)
% Computes the background sheared jet

    %% Compute the normalized pressure coordinate (Ullrich, 2015)
    pcoord = exp(lpref) / p0;
    lpcoord = log(pcoord);

    %% Compute the decay portion of the jet profile
    jetDecay = exp(-(lpcoord.^2 / prs.b^2));
    ujref = -prs.uj * (lpcoord .* jetDecay) + prs.u0;
    
    %% Compute the shear
    dujref = -prs.uj * jetDecay .* (1.0 - 2 * lpcoord.^2 / prs.b^2) ./ pcoord;
    dujref = (1.0 / p0) * dujref;
    dujref = (p0 * pcoord) .* dlpref .* dujref;
end

