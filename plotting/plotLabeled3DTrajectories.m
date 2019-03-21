function [h] = plotLabeled3DTrajectories(Data, est_states, titlename, labels)

h = figure('Color',[1 1 1]);
% vivid_cmap = vivid(length(labels));
vivid_cmap = hsv(length(labels));
for i=1:length(Data)
    
    % Extract data from each time-series    
    X = Data{i}';
    X3d= X(1:3,:);    
    
    % Plot Inferred Segments
    for ii=1:length(labels)
        est_labels = find(est_states{i}==labels(ii));
        plot3(X3d(1,est_labels),X3d(2,est_labels),X3d(3,est_labels),'.','Color',vivid_cmap(labels(ii),:),'LineWidth',1); hold on;
    end
    
end
xlabel('$x$','Interpreter','LaTex');ylabel('$y$','Interpreter','LaTex');zlabel('$z$','Interpreter','LaTex')

% Labels for colormap
colormap(vivid_cmap);
string_labels = [];
for l=1:length(labels)
    string_labels{l} = num2str(labels(l));
end
colorbar('YTick',linspace(0,1,length(labels)),'YTickLabel',string_labels);
title(titlename,'Interpreter','LaTex', 'FontSize',20)
grid on;


end