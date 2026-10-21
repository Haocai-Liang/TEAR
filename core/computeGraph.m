function [B, K, R] = computeGraph(clusterIndicator, order, weights)
% Precompute low-rank factorization matrices B, R, and K.

H = sparse(clusterIndicator);
[n, k] = size(H);
clusterSizes = sum(H, 1)';
degree = H * (weights .* clusterSizes);
degree(degree < eps) = eps;

DInvSqrt = spdiags(1 ./ sqrt(degree), 0, n, n);
sqrtOmega = spdiags(sqrt(weights), 0, k, k);

B = DInvSqrt * H * sqrtOmega;
R = full(B' * B);

K = zeros(k, k);
for t = 1:order
    K = K - 2 * (R ^ (t - 1)) + (R ^ (2 * t - 1));
end
end
