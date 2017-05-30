function[h] = plotTransMatrix(A)

h = figure('Color',[1 1 1]);
imagesc(A);
title ('Transition Matrix','Interpreter','LaTex')
level = 20; n = ceil(level/2);
cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
cmap = [cmap1; cmap2(2:end, :)];
colormap(vivid(cmap));
colorbar

end