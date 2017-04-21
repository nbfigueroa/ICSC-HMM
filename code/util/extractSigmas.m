function [Total_feats, sigmas] = extractSigmas(Data, bestGauPsi)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

max_feats = 0;
Zests = [];
iis = 1:Data.N;
for i=1:length(iis)
    ii = iis(i);
    Zest = bestGauPsi.stateSeq(1,ii).z;
    Zests = [Zests unique(Zest)];
    if  max(Zest)>max_feats
        max_feats = max(Zest);
    end
end

Total_feats = unique(Zests);

rec_thetas = bestGauPsi.theta(Total_feats);
sigmas = {};
for i = 1:length(rec_thetas)
    sigmas{i} = rec_thetas(i).invSigma^-1;
end


end

