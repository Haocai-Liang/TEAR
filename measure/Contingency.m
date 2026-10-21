function Cont = Contingency(Mem1, Mem2)
% Compute contingency matrix between two clustering partitions.
% Adapted from open-source code by David Corney (2000), implementing Hubert & Arabie (1985), "Comparing Partitions".

Cont = zeros(max(Mem1), max(Mem2));
for i = 1:length(Mem1)
   Cont(Mem1(i), Mem2(i)) = Cont(Mem1(i), Mem2(i)) + 1;
end
end
