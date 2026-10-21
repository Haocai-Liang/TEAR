function [X_cell, dims, originalDims] = adaGran(baseCls, c, tau, splitSeed)
% Adaptive cluster granulation.

maxLeaves = c;
[n, v] = size(baseCls);

normalizedBaseCls = zeros(n, v);
originalDims = zeros(v, 1);
groupsCell = cell(1, v);
HCell = cell(1, v);

for i = 1:v
    [~, ~, normalizedBaseCls(:, i)] = unique(baseCls(:, i));
    originalDims(i) = max(normalizedBaseCls(:, i));
    HCell{i} = sparse(normalizedBaseCls(:, i), 1:n, 1, originalDims(i), n);
    groupsCell{i} = accumarray(normalizedBaseCls(:, i), (1:n)', [], @(idx) {idx});
end

X_cell = cell(1, v);
dims = zeros(v, 1);

for i = 1:v
    other = [1:i-1, i+1:v];
    R = vertcat(HCell{other});
    Z = computeConf(HCell{i}, R);
    
    otherLabels = normalizedBaseCls(:, other);
    groups = groupsCell{i};
    splitLabels = zeros(n, 1);
    nextLabel = 0;
    
    for k = 1:originalDims(i)
        idx = groups{k};
        seed = mod(double(splitSeed) + 104729 * (i - 1) + 1009 * (k - 1), 2^32);
        labels = granulateBFSCluster(Z(idx, :), otherLabels(idx, :), maxLeaves, tau, seed);
        numSubLeaves = max(labels);
        
        splitLabels(idx) = nextLabel + labels;
        nextLabel = nextLabel + numSubLeaves;
    end
    
    dims(i) = nextLabel;
    X_cell{i} = sparse(splitLabels, 1:n, 1, dims(i), n);
end
end

function labels = granulateBFSCluster(Z, otherLabels, maxLeaves, tau, seed)
% Granulate a single cluster.
nCluster = size(Z, 1);
if nCluster <= 1 || maxLeaves <= 1
    labels = ones(nCluster, 1);
    return;
end

leafCount = 1;
queue = {struct('indices', (1:nCluster)', 'path', 'r')};
isLeaf = true;
head = 1;

while head <= numel(queue) && leafCount < maxLeaves
    current = queue{head};
    idx = current.indices;
    path = current.path;
    
    children = {};
    if numel(idx) >= 2 && leafCount < maxLeaves
        pathHash = 0;
        for ch = double(path)
            pathHash = mod(33 * pathHash + ch, 2^32);
        end
        rng(mod(double(seed) + pathHash, 2^32), 'twister');
        
        if size(unique(Z(idx, :), 'rows'), 1) >= 2
            z = kmeans(Z(idx, :), 2, 'MaxIter', 100, ...
                'Replicates', 1, 'Display', 'off', 'EmptyAction', 'singleton');
            child1 = idx(z == 1);
            child2 = idx(z == 2);
            if ~isempty(child1) && ~isempty(child2)
                score = splitScore(z, otherLabels(idx, :), tau);
                if score < 0
                    if min(child1) > min(child2)
                        [child1, child2] = deal(child2, child1);
                    end
                    children = {child1, child2};
                end
            end
        end
    end
    
    if ~isempty(children)
        isLeaf(head) = false;
        queue{end+1} = struct('indices', children{1}, 'path', [path '0']);
        queue{end+1} = struct('indices', children{2}, 'path', [path '1']);
        isLeaf = [isLeaf, true, true];
        leafCount = leafCount + 1;
    end
    head = head + 1;
end

leafIds = find(isLeaf);
leafPaths = cellfun(@(item) item.path, queue(leafIds), 'UniformOutput', false);
[~, order] = sort(leafPaths);
leafIds = leafIds(order);

labels = zeros(nCluster, 1);
for a = 1:numel(leafIds)
    labels(queue{leafIds(a)}.indices) = a;
end
end
