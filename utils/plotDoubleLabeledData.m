function [ ] = plotDoubleLabeledData( X, t , titlename, legends, label_range_z, label_range_s)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[D  N] = size(X);
if isempty(t)
    t = 1:N;
end

M = 10;
begin_time = t(1); 
xs = 1:N;    
ys = linspace(min(min(X(1:end-2,:))), max(max(X(1:end-2,:))),M);


% segment_labels = repmat(X(D,:), M, 1) ;
% cluster_labels = repmat(X(D-1,:), M/2, 1) + max(label_range_s);
% segment_labels(ceil(M/2)+1:end,:) = cluster_labels;


cluster_labels = repmat(X(D-1,:), M, 1) ;
segment_labels = repmat(X(D,:), M/2, 1) + max(label_range_z) ;
cluster_labels(1:ceil(M/2),:) = segment_labels;

imagesc( xs, ys, cluster_labels, [1 max(label_range_z)+max(label_range_s)] ); hold on;
% imagesc( xs, ys, cluster_labels ); hold on;

% level = 20; n = ceil(level/2);
% cmap1 = [linspace(1, 1, n); linspace(0, 1, n); linspace(0, 1, n)]';
% cmap2 = [linspace(1, 0, n); linspace(1, 0, n); linspace(1, 1, n)]';
% cmap = [cmap1; cmap2(2:end, :)];
% colormap(vivid(cmap, [.85, .85]));

% colormap(hsv)
% colormap(jet)
% cmap = hsv(max(label_range_s) + max(label_range_z));
% colormap(cmap);

cmap_clusters = hsv(max(label_range_z))
cmap_segments = hsv(max(label_range_z)+max(label_range_s))

custom_cmap = [cmap_clusters; cmap_segments(5:end,:)];
colormap(custom_cmap);

alpha(0.2)

% plot data
plot(t - begin_time, X(1:end-2,:)','-.', 'LineWidth',2); hold on;

% plot separating line
plot(repmat(mean(ys),1,length(xs)),'-.', 'LineWidth',1, 'Color', [0 0 1]); hold on;


axis( [1 N ys(1) ys(end)] );
xlabel('Time (1,...,T)','Interpreter','LaTex')
ylabel('$\mathbf{x}_t$','Interpreter','LaTex')
if ~isempty(legends)
    legend(legends);
end
set(gca,'YDir','normal')
title(titlename,'Interpreter','LaTex','Fontsize',16)

end

