function [S] = matrixExp(S)
[V,D] = eig(S);
S = V*(diag(exp(diag(D))))*V';
end