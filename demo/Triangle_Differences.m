%% Physically possible triangles %%%%%%%%
clear all
clc
anglesC = [];
for alpha=1:180
    for beta=1:(179-alpha)
        gamma = 180 - alpha - beta; 
        anglesC{alpha+1,beta+1} = [alpha;beta;gamma]; 
    end    
end
angles_tmp = reshape(anglesC,1,prod(size(anglesC)));
angles_tmp = angles_tmp(~cellfun(@isempty, angles_tmp));
angles = cell2mat(angles_tmp)*pi/180; 
%% Triangle differences
frob_norms = [];
cos_sims = [];
tic
% for a=1:length(angles)
for a=1:length(angles)
    for i=1:length(angles)
        xi = sort(angles(:,a));
        xj = sort(angles(:,i));
        %Frobenius Norm of the Cosines
        frob_norms(a,i) = norm(cos(xi)-cos(xj),'fro');
        %Cosine Simlarity of the Angles
        cos_sims(a,i) = (xi'*xj)/(norm(xi)*norm(xj));
    end
end
toc
%% Visualize
figure('Color',[1 1 1])
% subplot(2,1,1)
% diff = frob_norms;
% diff_norm = (diff - min(min(diff)))/(max(max(diff)) - min(min((diff))));
% imshow(diff_norm, 'InitialMagnification',1000)  % # you want your cells to be larger than single pixels
% title('Frobenius Difference of Eigenspace Angles')
% colormap(jet)
% colorbar
% 
% subplot(2,1,2)
imshow(cos_sims, 'InitialMagnification',1000)  % # you want your cells to be larger than single pixels
title('Cosine Similarity of EigenSpace Angles')
colormap(jet)
colorbar