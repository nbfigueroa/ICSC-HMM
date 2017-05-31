function[h] = plotTransMatrix(A, varargin)

if isempty(varargin)
    titlename = 'Transition Matrix';
    plot_figure  = 1;
    label_range = 1:length(A);
else
    titlename = varargin{1};
    plot_figure  = varargin{2};
    label_range  = varargin{3};
end

if plot_figure
    h = figure('Color',[1 1 1]);
end

imagesc(A);

xticklabels = label_range;
xticks = linspace(1, size(A, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)

yticklabels = label_range;
yticks = linspace(1, size(A, 2), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels)

title ({titlename},'Interpreter','LaTex')
level = 20; n = ceil(level/2);
cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
cmap = [cmap1; cmap2(2:end, :)];
colormap(vivid(cmap));
colorbar


end