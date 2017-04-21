function [S] = matrixSquareRoot(S)
[V,D] = eig(S);
S = V*(D^(1/2))*V';
end