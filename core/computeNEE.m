function weights = computeNEE(baseCls, fragmentIndicator, theta, normalizers)
% Compute granule confidence weights using Normalized Ensemble Entropy.
% Adapted from the open-source implementation in SDGCA (Xu Zhang et al., 2025).

[n, v] = size(baseCls);
k = size(fragmentIndicator, 1);

clusterCounts = zeros(v, 1);
normalizedBase = zeros(n, v);
for j = 1:v
    [~, ~, normalizedBase(:, j)] = unique(baseCls(:, j));
    clusterCounts(j) = max(normalizedBase(:, j));
end

offsets = [0; cumsum(clusterCounts)];
totalOrigClusters = offsets(end);
refRows = zeros(n * v, 1);
refCols = zeros(n * v, 1);
idx = 1;
for j = 1:v
    cols = (1:n)';
    rows = normalizedBase(:, j) + offsets(j);
    refRows(idx:idx+n-1) = rows;
    refCols(idx:idx+n-1) = cols;
    idx = idx + n;
end
refIndicator = sparse(refRows, refCols, 1, totalOrigClusters, n);

membership = spones(sparse(fragmentIndicator));
Q = membership * refIndicator';
sizes = full(sum(membership, 2));

entropyValues = zeros(k, 1);
active = (sizes > 1) & (normalizers(:) > 1);

for j = 1:v
    cols = (offsets(j) + 1):offsets(j + 1);
    [rows, ~, counts] = find(Q(:, cols));
    rows = rows(:);
    counts = counts(:);
    prob = counts ./ sizes(rows);
    terms = prob .* log2(prob + eps);
    entropyValues = entropyValues - accumarray(rows, terms, [k, 1]);
end

entropyValues(active) = entropyValues(active) ./ log2(normalizers(active));
entropyValues(~active) = 0;

weights = exp(-entropyValues ./ (theta * v));
end
