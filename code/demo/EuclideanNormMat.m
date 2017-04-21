function [d] = EuclideanNormMat(X)
% Euclidean distance of 
d = sqrt(trace(X'*X));
end
