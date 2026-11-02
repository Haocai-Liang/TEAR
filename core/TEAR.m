function [labels, model] = TEAR(baseCls, c, options)
% Linear-Time Ensemble Clustering with Adaptive Self-Refinement of Clusters.

baseCls = double(baseCls);
[n, v] = size(baseCls);
for p = 1:v
    [~, ~, baseCls(:, p)] = unique(baseCls(:, p));
end

tau = options.tau;
lambda = options.lambda;
theta = options.theta;
o = options.o;
d = c;
l = c;

splitSeed = options.seed + 100000 + options.trial;
solverSeed = options.seed + 200000 + options.trial;
predSeed = options.seed + 300000 + options.trial;

% 1. Adaptive cluster granulation
[X_cell, dims, originalDims] = adaGran(baseCls, c, tau, splitSeed);

% 2. Granule confidence weighting
fragmentIndicator = vertcat(X_cell{:});
fragmentNormalizers = repelem(originalDims, dims);
weights = computeNEE(baseCls, fragmentIndicator, theta, fragmentNormalizers);

% 3. Low-rank graph precomputation
[B, K, R] = computeGraph(fragmentIndicator', o, weights);
graph = struct('B', B, 'K', K, 'R', R, 'o', o);

% 4. Alternating optimization
problem = struct('X', {X_cell}, 'dims', dims, 'numBase', v, 'n', n, 'd', d, 'l', l);
solverOpts = struct('lambda', lambda, 'maxIter', 50, 'tol', 1e-2, 'initSeed', solverSeed);
model = solver_TEAR(problem, graph, solverOpts);

% 5. Label extraction
labels = getLabels(model.Y, c, predSeed);
end
