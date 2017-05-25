function [S] = matrixLog(S)
[V,D] = eig(S);
S = V*(diag(log(diag(D))))*V';
end