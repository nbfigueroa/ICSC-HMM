function [ New_POS ] = interpolateSpline(POS,min_length )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
dim = size(POS,1);
    for kk=1:dim            
        % Make original data
        oldPcntVals = linspace(0,1,length(POS));
        oldYvals = POS(kk,:);
        % Set a new spacing from 0 to 1 and interpolate
        newPcntVals = linspace(0,1,min_length);
        newYvals = interp1(oldPcntVals, oldYvals, newPcntVals,'spline');            
        New_POS(kk,:) = newYvals;
    end
end

