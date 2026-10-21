function model = solver_TEAR(problem, graph, opts)
% Alternating optimization for W^(p), A, and Y.

X = problem.X;
v = problem.numBase;
n = problem.n;
d = problem.d;
l = problem.l;

rng(opts.initSeed, 'twister');

A = orthogonalFactor(rand(d, l));
Y = ones(l, n) ./ l;
W = cell(1, v);
for p = 1:v
    W{p} = orthogonalFactor(rand(problem.dims(p), d));
end

graph.yAlpha = v + opts.lambda * graph.o;
graph.yKernel = (graph.yAlpha * eye(size(graph.R, 1)) + opts.lambda * graph.K * graph.R) \ graph.K;

for iter = 1:opts.maxIter
    previous.W = W;
    previous.A = A;
    previous.Y = Y;
    
    % optimize W
    Dsum = zeros(d, n);
    for p = 1:v
        C = X{p} * Y' * A';
        W{p} = orthogonalFactor(C);
        Dsum = Dsum + W{p}' * X{p};
    end
    
    % optimize A
    A = orthogonalFactor(Dsum * Y');
    
    % optimize Y
    C = A' * Dsum;
    CB = C * graph.B;
    Y = (C / graph.yAlpha) - (opts.lambda / graph.yAlpha) * (CB * graph.yKernel) * graph.B';
    
    % Convergence check on variable changes
    diffW = maxCellDiff(W, previous.W);
    diffA = relativeDiff(A, previous.A);
    diffY = relativeDiff(Y, previous.Y);
    if max([diffW, diffA, diffY]) < opts.tol
        break;
    end
end

model.W = W;
model.A = A;
model.Y = Y;
end

function Q = orthogonalFactor(M)
[L, ~, R] = svd(M, 'econ');
Q = L * R';
end

function value = maxCellDiff(current, previous)
value = 0;
for p = 1:numel(current)
    value = max(value, relativeDiff(current{p}, previous{p}));
end
end

function value = relativeDiff(current, previous)
denom = max(1, norm(previous, 'fro'));
value = norm(current - previous, 'fro') / denom;
end
