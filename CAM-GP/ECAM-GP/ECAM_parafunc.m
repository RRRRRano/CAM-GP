function [rmse, nse] = ECAM_parafunc(tradata, testdata, p, q, m, tau, ...
                                    lambda1, lambda2, maxIter, n, n2, rcoord)
%ECAM_PARAFUNC  Fit ECAM-GP by penalised maximum likelihood and evaluate it.
%
%   [RMSE, NSE] = ECAM_PARAFUNC(TRADATA, TESTDATA, P, Q, M, TAU, LAMBDA1,
%   LAMBDA2, MAXITER, N, N2, RCOORD)
%
%   Same interface as CAM_parafunc.  ECAM-GP shares one length scale across the
%   continuous variables within each adjustment term and fixes the length scale
%   of the first level of every categorical variable at one, so that the number
%   of adjustment length scales is sum(m) - q instead of p*sum(m).

    dat  = tradata(:, 1:(p+q));
    y    = tradata(:, p+q+1);
    ndat = testdata(:, 1:(p+q));
    ny   = testdata(:, p+q+1);

    % Recode the categorical levels to 1..m(h).
    for h = 1:q
        col = p + h;
        [~, ~, ic] = unique(dat(:,col));
        dat(:,col) = ic;
        [~, ~, icTest] = unique(ndat(:,col));
        ndat(:,col) = icTest;
    end

    % Parameter vector layout (see ECAM_covx_add for details):
    %   2 + q           variance parameters
    %   2p              baseline length scales
    %   sum(m) - q      adjustment length scales, one per level and variable,
    %                   levels 2..m(h)
    %   1               mixing weight
    nVariance = 2 + q;
    nAdjScale = sum(m) - q;

    lb = [0.1 * ones(nVariance, 1);
          0.1 * ones(2*p, 1);
          0.1 * ones(nAdjScale, 1);
          -10];                        % alpha_raw, unconstrained
    ub = [100 * ones(nVariance, 1);
          10  * ones(2*p, 1);
          10  * ones(nAdjScale, 1);
          10];

    x0 = (lb + ub) / 2;
    x0(end) = 0;

    % Unconstrained quasi-Newton optimiser, as used for the EEzGP baseline.
    options = optimoptions('fminunc', ...
        'Algorithm',             'quasi-newton', ...
        'TolX',                  1e-6, ...
        'MaxFunctionEvaluations', maxIter, ...
        'Display',               'iter');

    objfun = @(x) ECAM_eval_f_list(x, dat, y, p, q, m, tau, n, rcoord, ...
                                   lambda1, lambda2);
    solpar = fminunc(objfun, x0, options);

    % Test-set prediction.
    rmse = ECAM_rmse(solpar, ndat, ny, dat, y, rcoord, p, q, tau, m, n);

    % Nash-Sutcliffe efficiency, 1 - sum((yhat-y)^2) / sum((y-ymean)^2).
    nse = 1 - n2 * rmse^2 / (var(ny) * (n2 - 1));
end
