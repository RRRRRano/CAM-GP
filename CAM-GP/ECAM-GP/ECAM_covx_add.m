function res = ECAM_covx_add(w1, w2, parv, p, q, m)
%ECAM_COVX_ADD  Covariance between two inputs under ECAM-GP.
%
%   RES = ECAM_COVX_ADD(W1, W2, PARV, P, Q, M)
%
%   W1 and W2 are 1-by-(p+q) inputs, ordered as continuous variables followed by
%   categorical levels coded 1..m(h).
%
%   ECAM-GP is the efficient variant of CAM-GP.  Within each adjustment term the
%   additive and multiplicative components use a single length scale shared by
%   all continuous variables, and the length scale of the first level of each
%   categorical variable is fixed at one for identifiability.
%
%   PARV holds the model parameters:
%     parv(1)                       sigma_0a^2  baseline additive variance
%     parv(2)                       sigma_0m^2  baseline multiplicative variance
%     parv(3 : 2+q)                 sigma_h^2   adjustment-term variances
%     parv(3+q : 2+q+p)             theta_0a    baseline additive length scales
%     parv(3+q+p : 2+q+2p)          theta_0m    baseline multiplicative length scales
%     parv(2+q+2p+1 : end-1)        adjustment length scales, levels 2..m(h)
%     parv(end)                     alpha_raw   unconstrained mixing weight

    % transpose so that x1 and x2 are p-by-1 column vectors: this keeps
    % theta0a .* diffSq an elementwise operation on p-by-1 vectors
    x1 = w1(1:p).';
    x2 = w2(1:p).';

    diffSq    = (x1 - x2).^2;
    sumDiffSq = sum(diffSq);

    % ---- baseline kernel, Eq. 21 ----------------------------------------
    sigma0a = parv(1);
    sigma0m = parv(2);
    theta0a = parv(3+q   : 2+q+p);
    theta0m = parv(3+q+p : 2+q+2*p);

    res = sigma0a * mean(exp(-theta0a .* diffSq)) ...
        + sigma0m * exp(-sum(theta0m .* diffSq));

    % ---- mixing weight, alpha in (0,1) ----------------------------------
    alpha = 1 / (1 + exp(-parv(end)));

    % ---- adjustment terms ------------------------------------------------
    idxAdj = 2 + q + 2*p + 1;          % first adjustment length scale

    for h = 1:q
        sigmaH = parv(2 + h);

        if w1(p+h) ~= w2(p+h)
            continue                        % levels differ: no contribution
        end

        l = w1(p+h);
        if l == 1
            thetaVal = 1;                   % first level: fixed at one
        else
            offset   = sum(m(1:h-1)) - (h-1);
            thetaVal = parv(idxAdj + offset + (l - 2));
        end

        addPart  = mean(exp(-thetaVal .* diffSq));
        multPart = exp(-thetaVal * sumDiffSq);

        res = res + sigmaH * (alpha * addPart + (1 - alpha) * multPart);
    end
end
