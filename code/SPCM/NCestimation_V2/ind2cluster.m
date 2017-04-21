function [clusters, newlabels] = ind2cluster(labels)

C = unique(labels);
newlabels = labels;
k = length(C);
clusters = cell(1,k);
for i = 1:k
  ind = find(labels==C(i));
  clusters{i} = ind;
  newlabels(ind) = i;
end