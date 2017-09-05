function [ujref,dujref] = computeJetProfileUniform(prs, lpref)
% Computes the background sheared jet

    ujref = prs.u0 * ones(size(lpref));
    dujref = 0.0 * ujref;
end

