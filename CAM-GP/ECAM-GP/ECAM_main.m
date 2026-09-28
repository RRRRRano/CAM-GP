%% ECAM_main.m
%  ECAM-GP: efficient variant of CAM-GP for surrogate modelling with mixed
%  categorical and continuous inputs.
%
%  ECAM-GP shares a single length scale across the continuous variables within
%  each adjustment term and fixes the length scale of the first level of every
%  categorical variable at one, which keeps the number of parameters close to
%  that of EzGP.
%
%  Fits ECAM-GP to one of the case studies reported in the paper and prints the
%  test root-mean-square error (RMSE) and the Nash-Sutcliffe efficiency (NSE)
%  for a single training set (NR = 1).
%
%  Set DATASET below to select the case study.
%
%  Reference
%    L. Wang, X. Duan, J. Hu, Z. Liu, L. Yan.
%    A composite additive-multiplicative Gaussian process surrogate for mixed
%    categorical and continuous design variables. Computers & Structures.
%
%  Requirements
%    MATLAB R2024a or later, Optimization Toolbox (fminunc).

clear; clc; close all;

%% ------------------------- select the case study ------------------------
DATASET = 1;
%  1 : Example 1             3 continuous, 3 categorical     81 train /  1215 test
%  2 : Example 2             9 continuous, 9 categorical    243 train /  1215 test
%  3 : Beam bending          2 continuous, 1 categorical     60 train / 10000 test
%  4 : Embankment            1 continuous, 3 categorical    200 train /    29 test
%  5 : Embankment extended   3 continuous, 3 categorical    200 train /    29 test
%      (the extended data contain two continuous variables that are irrelevant
%       to the response; only the first one is physically meaningful)

%% ------------------------- case-study settings --------------------------
% tau        numerical stabiliser added to the diagonal of the covariance matrix
% lambda1    penalty on the variance parameters
% lambda2    penalty on the length-scale parameters
% maxIter    maximum number of objective evaluations of the optimiser
switch DATASET
    case 1
        dataFile = 'example1.mat';
        tau = 1;      lambda1 = 0.05; lambda2 = 0.5; maxIter = 100;
    case 2
        dataFile = 'example2.mat';
        tau = 50;     lambda1 = 0.05; lambda2 = 0.5; maxIter = 400;
    case 3
        dataFile = 'beam_bending.mat';
        tau = 1e-3;   lambda1 = 0.05; lambda2 = 0.5; maxIter = 100;
    case 4
        dataFile = 'embankment.mat';
        tau = 20;     lambda1 = 0.05; lambda2 = 0.5; maxIter = 100;
    case 5
        dataFile = 'embankment_extended.mat';
        tau = 30;     lambda1 = 0.05; lambda2 = 0.5; maxIter = 100;
    otherwise
        error('ECAM_main:badDataset', 'DATASET must be an integer between 1 and 5.');
end

%% ------------------------- locate and load the data ---------------------
% The dataset folder is a sibling of the CAM-GP and ECAM-GP folders.  The
% script may be run from the repository root, from inside ECAM-GP/, or after the
% folder has been added to the MATLAB path, so several candidate locations are
% tried before giving up.
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end

candidates = { fullfile(thisDir, '..', 'dataset'), ...
               fullfile(thisDir, 'dataset'), ...
               fullfile(thisDir, '..', '..', 'dataset'), ...
               fullfile(pwd,     'dataset'), ...
               fullfile(pwd,     '..', 'dataset') };

% The location of the companion functions on the MATLAB path is a more reliable
% anchor than mfilename, because a script that is run with unsaved changes is
% executed from a temporary copy.
fncDir = fileparts(which('ECAM_covx_add'));
if ~isempty(fncDir)
    candidates = [ { fullfile(fncDir, '..', 'dataset'), ...
                     fullfile(fncDir, 'dataset') }, candidates ];
end

dataDir = '';
for k = 1:numel(candidates)
    if exist(fullfile(candidates{k}, dataFile), 'file')
        dataDir = candidates{k};
        break
    end
end

if isempty(dataDir)
    error('ECAM_main:dataNotFound', ...
        ['Cannot find "%s".\n\n' ...
         'Run this script from the repository that contains the "dataset", ' ...
         '"CAM-GP" and "ECAM-GP" folders.\n' ...
         'If you edited DATASET, save the file before running it: MATLAB ' ...
         'otherwise executes a temporary copy of the script, and the dataset ' ...
         'folder cannot be located.\n\nSearched in:\n  %s'], ...
        dataFile, strjoin(candidates, sprintf('\n  ')));
end

S  = load(fullfile(dataDir, dataFile));
p  = S.p;
q  = S.q;
m  = S.m;
n  = size(S.xtr, 1);
n2 = size(S.xte, 1);

% Input layout expected by the kernel functions:
%   columns 1..p        continuous variables
%   columns p+1..p+q    categorical levels, coded 1..m(h)
%   column  p+q+1       response
tradata  = [S.xtr, S.ztr, S.ytr];
testdata = [S.xte, S.zte, S.yte];

%% ------------------------- index pairs of the upper triangle ------------
colIdxCellArray = cell(1, n-1);
for nn = (n-1):-1:1
    colIdxCellArray{n-nn} = (1:nn)' + n - nn;
end
colIdx = vertcat(colIdxCellArray{:});
rowIdx = repelem((1:n-1)', (n-1):-1:1);
rcoord = [rowIdx, colIdx];

%% ------------------------- fit and evaluate -----------------------------
fprintf('Case study   : %s\n', S.caseName);
fprintf('Data file    : %s\n', dataFile);
fprintf('Inputs       : p = %d continuous, q = %d categorical, m = [%s]\n', ...
        p, q, num2str(m));
fprintf('Training set : %d samples\n', n);
fprintf('Test set     : %d samples\n', n2);
fprintf('Settings     : tau = %g, lambda1 = %g, lambda2 = %g, maxIter = %d\n\n', ...
        tau, lambda1, lambda2, maxIter);

tic;
[rmse, nse] = ECAM_parafunc(tradata, testdata, p, q, m, tau, ...
                            lambda1, lambda2, maxIter, n, n2, rcoord);
elapsed = toc;

fprintf('\n--------------------------------------------\n');
fprintf('  RMSE  = %.6f\n', rmse);
fprintf('  NSE   = %.6f\n', nse);
fprintf('  Time  = %.2f s\n', elapsed);
fprintf('--------------------------------------------\n');
