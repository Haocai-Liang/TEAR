clear; clc;
addpath('./core');
addpath('./measure');
addpath('./datasets');

datasetName = 'D06-Optdigits.mat'; % Change dataset name here
tau = 0.1;                 % Granulation parameter tau
lambda = 1e2;              % Regularization parameter lambda

load(datasetName, 'members', 'gt');
[~, poolSize] = size(members);
c = numel(unique(gt));
v = 20;
numTrials = 10;

rng(2026);
selections = zeros(numTrials, v);
for trial = 1:numTrials
    p = randperm(poolSize);
    selections(trial, :) = p(1:v);
end

ACC = zeros(numTrials, 1);
NMI = zeros(numTrials, 1);

for trial = 1:numTrials
    baseCls = members(:, selections(trial, :));
    opts = struct('tau', tau, 'lambda', lambda, 'theta', 0.1, 'o', 3, ...
        'seed', 2026, 'trial', trial);
    pred = TEAR(baseCls, c, opts);
    scores = Clustering8Measure(gt, pred);
    
    ACC(trial) = scores(1);
    NMI(trial) = scores(2);
    
    trial_num = trial;
    trial_acc = ACC(trial);
    trial_nmi = NMI(trial);
end

mean_acc = mean(ACC)
std_acc = std(ACC)
mean_nmi = mean(NMI)
std_nmi = std(NMI)
