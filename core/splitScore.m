function score = splitScore(labels, otherLabels, tau)
% Evaluate candidate cluster split via sparse contingency tables.

n = numel(labels);
sizes = accumarray(labels, 1);
if numel(sizes) < 2
    score = inf;
    crossSupport = 0;
    cutPairs = 0;
    return;
end

cutPairs = (n * (n - 1) - sum(sizes .* (sizes - 1))) / 2;

crossSupport = 0;
numOther = size(otherLabels, 2);

for j = 1:numOther
    counts = sparse(otherLabels(:, j), labels, 1);
    totals = full(sum(counts, 2));
    inside = nonzeros(counts);
    cutInJ = (sum(totals .* (totals - 1)) - sum(inside .* (inside - 1))) / 2;
    crossSupport = crossSupport + cutInJ;
end

score = 2 * crossSupport - tau * numOther * cutPairs;
end
