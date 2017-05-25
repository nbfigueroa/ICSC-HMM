function [ H ] = convert2H( X )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


if size(X,2) > 1
    H = zeros(4,4,length(X));
    H(1:3,1:3,1:end) = reshape(quaternion(X(4:7,1:end)),[3 3 length(X)]);
    H(1:3,4,1:end)   = reshape(X(1:3,1:end),[3 1 length(X)]);
else
    H = zeros(4,4);
    H(1:3,1:3) = quaternion(X(4:7));
    H(1:3,4)   = X(1:3);
end
end

