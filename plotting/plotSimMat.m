function [h] = plotSimMat( S )
h = figure('Color',[1 1 1]);
imagesc(S);
set( gca, 'FontSize', 16);
% 
% level = 10; n = ceil(level/2);
% cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
% cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
% cmap = [cmap1; cmap2(2:end, :)];
% colormap(vivid(cmap, [0.85 0.85]));

colormap(bone);
colorbar
title ('Similarity Matrix of Emission Models','Interpreter','LaTex')
grid on


end