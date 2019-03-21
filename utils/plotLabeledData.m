function [ ] = plotLabeledData( X, t , titlename, legends, label_range)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D  N] = size(X);

if isempty(t)
    t = 1:N;
end

M = 10;
begin_time = t(1); 
xs = 1:N;    
ys = linspace(min(min(X(1:end,:))), max(max(X(1:end-1,:))),M);

% Without label_range
if isempty(label_range)
    segment_labels = X(D,:);
    imagesc(xs,ys,segment_labels,'CDataMapping','scaled'); hold on;
else  % With label_range  
    segment_labels = repmat(X(D,:), M, 1);
    imagesc( xs, ys, segment_labels, [1 max( label_range)] ); hold on;
end

% level = 20; n = ceil(level/2);
% cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
% cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
% cmap = [cmap1; cmap2(2:end, :)];
% colormap(vivid(cmap, [.85, .85]));

cmap = hsv(length(label_range));
colormap(cmap);
alpha 0.2;

plot(t - begin_time, X(1:end-1,:)','-.', 'LineWidth',2);hold on;
axis( [1 N ys(1) ys(end)] );
xlabel('Time (1,...,T)','Interpreter','LaTex')
ylabel('$\mathbf{x}_t$','Interpreter','LaTex','Fontsize',16)

if ~isempty(legends)
    legend(legends,'Interpreter','LaTex','Fontsize',12)
end

set(gca,'YDir','normal')
title(titlename,'Interpreter','LaTex','Fontsize',16)

end

