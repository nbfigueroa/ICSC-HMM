function [handle] = plot3D_labeledTrajectories(Data, states, sub_sample, title_name, title_edge, set_view, varargin)

bimanual = 0;
if nargin == 7
    dim = varargin{1};
    if strcmp(dim,'robots')
        fprintf('here');
        bimanual = 1;
    end
end

clc;
X_data = []; labels_data = [];
for i=1:length(Data)    
    if bimanual
        X_data = [X_data; Data{i}(:,1:3); Data{i}(:,14:16)];
        labels_data = [labels_data; states{i}; states{i} ];
    else
        X_data = [X_data; Data{i}(:,1:3)];
        labels_data = [labels_data; states{i}];
    end
end

plot_options        = [];
plot_options.labels = labels_data(1:sub_sample:end);
plot_options.title  = [];
plot_options.points_size = 20;
handle = ml_plot_data(X_data(1:sub_sample:end,:),plot_options);
axis equal;
view(set_view)

% Create title
title(title_name,...
    'BackgroundColor',[1 1 1],...
    'EdgeColor',title_edge,...
    'FontSize',18,...
    'FontName','Times',...
    'Interpreter','latex');
end