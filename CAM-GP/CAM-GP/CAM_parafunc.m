function [rmse, nse] = CAM_parafunc(tradata, testdata, p, q, m, tau, ...
                                   lambda1, lambda2, maxIter, n, n2, rcoord)
%CAM_PARAFUNC  Fit CAM-GP by penalised maximum likelihood and evaluate it.
%
%   [RMSE, NSE] = CAM_PARAFUNC(TRADATA, TESTDATA, P, Q, M, TAU, LAMBDA1,
%   LAMBDA2, MAXITER, N, N2, RCOORD)
%
%   TRADATA is n-by-(p+q+1) and TESTDATA is n2-by-(p+q+1).  The columns are
%   ordered as continuous variables, categorical levels (coded 1..m(h)) and the
%   response.
%
%   TAU       numerical stabiliser added to the diagonal of the covariance
%             matrix
%   LAMBDA1   penalty applied to the variance parameters
%   LAMBDA2   penalty applied to the length-scale parameters
%   MAXITER   maximum number of objective evaluations
%   RCOORD    upper-triangular index pairs of the n-by-n covariance matrix
%
%   The parameters are estimated by minimising the penalised negative
%   log-likelihood of CAM_eval_f_list.  RMSE and NSE are evaluated on TESTDATA
%   with the constant mean estimated by generalised least squares.

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

    % Parameter vector layout (see CAM_covx_add for details):
    %   2 + q     variance parameters
    %   2p        baseline length scales
    %   p*sum(m)  adjustment length scales (p per level)
    %   1         mixing weight
    nVariance = 2 + q;

    lb = [0.01 * ones(nVariance, 1);
          0.01 * ones(2*p, 1);
          0.01 * ones(p*sum(m), 1);
          -10];                        % alpha_raw, unconstrained
    ub = [100 * ones(nVariance, 1);
          10  * ones(2*p, 1);
          10  * ones(p*sum(m), 1);
          10];

    x0 = (lb + ub) / 2;
    x0(end) = 0;                      

    options = optimoptions('fmincon', ...
        'Algorithm',            'sqp', ...
        'TolX',                 1e-8, ...
        'MaxFunctionEvaluations', maxIter, ...
        'FiniteDifferenceType', 'central', ...
        'Display',              'iter');

    objfun = @(x) CAM_eval_f_list(x, dat, y, p, q, m, tau, n, rcoord, ...
                                  lambda1, lambda2);
    solpar = fmincon(objfun, x0, [], [], [], [], lb, ub, [], options);

    % Test-set prediction.
    rmse = CAM_rmse(solpar, ndat, ny, dat, y, rcoord, p, q, tau, m, n);

    % Nash-Sutcliffe efficiency, 1 - sum((yhat-y)^2) / sum((y-ymean)^2).
    % Since sum((yhat-y)^2) = n2*rmse^2 and sum((y-ymean)^2) = (n2-1)*var(y),
    nse = 1 - n2 * rmse^2 / (var(ny) * (n2 - 1));
end
