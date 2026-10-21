function Z = computeConf(H, R)
% Compute external consensus features for a base clustering.

F = full(R' * (R * H'));
rowNorm = sqrt(sum(F.^2, 2));
nonzero = rowNorm > 0;
Z = F;
Z(nonzero, :) = Z(nonzero, :) ./ rowNorm(nonzero);
end
