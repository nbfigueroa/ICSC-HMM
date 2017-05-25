function [ X ] = convert2X( H )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
dim = size(H,3);
X(1:3,:) = reshape(H(1:3,end,:),[3 dim]);
X(4:7,:) = reshape(quaternion(H(1:3,1:3,:)),[4 dim]);

end

