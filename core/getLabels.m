function labels = getLabels(Y, c, predSeed)
% Extract consensus cluster labels from representation matrix Y.

% Column-wise stable softmax
Y = exp(Y - max(Y, [], 1));
Y = Y ./ sum(Y, 1);

% Economic SVD to obtain sample embeddings
[embedding, ~, ~] = svd(Y', 'econ');
if size(embedding, 2) > c
    embedding = embedding(:, 1:c);
end

% Row-wise L2 normalization
rowNorm = sqrt(sum(embedding.^2, 2));
rowNorm(rowNorm == 0) = 1;
embedding = embedding ./ rowNorm;

rng(predSeed, 'twister');

labels = kmeans(embedding, c, ...
    'MaxIter', 100, ...
    'Replicates', 3, ...
    'Display', 'off');
end
