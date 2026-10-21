function [nmi, clust_ent] = compute_nmi(T, H)
% Compute Normalized Mutual Information (NMI) between two clusterings.
% Adapted from A. Strehl and J. Ghosh, "Cluster Ensembles - A Knowledge Reuse Framework for Combining Multiple Partitions", JMLR 2002.
    
    N = length(T);
    classes = unique(T);
    clusters = unique(H);
    num_class = length(classes);
    num_clust = length(clusters);

    for j = 1:num_class
        index_class = (T(:) == classes(j));
        D(j) = sum(index_class);
    end      
    
    mi = 0;
    A = zeros(num_clust, num_class);
    avgent = 0;
    for i = 1:num_clust
        index_clust = (H(:) == clusters(i));
        B(i) = sum(index_clust);
        for j = 1:num_class
            index_class = (T(:) == classes(j));
            A(i, j) = sum(index_class .* index_clust);
            if (A(i, j) ~= 0)
                miarr(i, j) = A(i, j) / N * log2(N * A(i, j) / (B(i) * D(j)));
                avgent = avgent - (B(i) / N) * (A(i, j) / B(i)) * log2(A(i, j) / B(i));
            else
                miarr(i, j) = 0;
            end
            mi = mi + miarr(i, j);
        end        
    end
    
    class_ent = 0;
    for i = 1:num_class
        class_ent = class_ent + D(i) / N * log2(N / D(i));
    end
    
    clust_ent = 0;
    for i = 1:num_clust
        clust_ent = clust_ent + B(i) / N * log2(N / B(i));
    end
        
    nmi = 2 * mi / (clust_ent + class_ent);
end
