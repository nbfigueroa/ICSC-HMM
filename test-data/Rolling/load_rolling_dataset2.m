function [data, TruePsi, Data, True_states, data_legends, dataset_name, plot_subsample, dataset_view] = load_rolling_dataset2(data_path, time_series, do_plots, variables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Load One of the Wiping Datasets    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
do_plot_ind = 0;

dataset_name   = 'Task 3: Dough Rolling';
plot_subsample = 5; 
dataset_view = [102 16];

load(strcat(data_path,'Rolling/proc-data-labeled-real.mat'))

switch variables
    case 0
        weights = [ones(3,1); 10*ones(3,1); ones(4,1); -1/10*ones(3,1); -1/10*ones(3,1)];
        data_legends = {'$x_1$','$x_2$','$x_3$', ...
            '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$', ...
            '$q_i$','$q_j$','$q_k$','$q_w$', ...
            '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};        
    case 1
        weights = [ones(3,1); 10*ones(3,1); -1/10*ones(3,1); -1/10*ones(3,1)];
        data_legends = {'$x_1$','$x_2$','$x_3$', ...
            '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$', ...
            '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};        
    case 2
        weights = [ones(3,1); 1/50*ones(3,1); 1/5*ones(3,1)];
        data_legends = {'$x_1$','$x_2$','$x_3$', '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};        
    case 3
        weights = [ones(3,1); 5*ones(3,1)];
        data_legends = {'$x_1$','$x_2$','$x_3$', '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$'};
end

% Select Specific Time Series
Data(3) = [];
True_states(3) = [];

Data(time_series+1:end) = [];
True_states(time_series+1:end) = [];

% At this point I have Data{..} and True_states{..}
Data_orig = Data;
Data_new = []; dt = 1/100; window_size = 13; 
crop_size = (window_size+1)/2;
% Compute Velocities and Euler Angle Representation
for i=1:length(Data_orig)
    clear X3d quat_tmp ft_mp

    % Create velocity vectors and crop time-series
    if true
        dx_nth = sgolay_time_derivatives(Data_orig{i}(1:3,:)', dt, 2, 3, window_size);
        X3d     = dx_nth(:,:,1)';
        X3d_dot = dx_nth(:,:,2)';
        quat_tmp = Data_orig{i}(4:7,crop_size:end-crop_size);
        ft_tmp   = Data_orig{i}(8:end,crop_size:end-crop_size);
        True_states{i} = True_states{i}(1,crop_size:end-crop_size);
    else        
        X3d     = Data_orig{i}(1:3,:);
        X3d_dot = [zeros(3,1) diff(X3d')'];
        quat_tmp = Data_orig{i}(4:7,:);
        ft_tmp   = Data_orig{i}(8:end,:);        
    end
    
    % Create euler angles
    xyz = zeros(3,length(quat_tmp));
    for ii=1:length(xyz)
        R_tmp = quaternion2matrix(quat_tmp(:,ii));
        xyz(:,ii) = R2rpy(R_tmp);
    end   
    
    % Fill in new data structure
    Data_new{i}.X     = X3d;
    Data_new{i}.X_dot = X3d_dot;
    Data_new{i}.euler = xyz;
    Data_new{i}.quat = quat_tmp;
    Data_new{i}.ft    = ft_tmp;
end

N = length(Data_orig);
Data   = [];
for ii=1:N
    switch variables
        case 0
            Data{ii}  = [Data_new{ii}.X; Data_new{ii}.X_dot; Data_new{ii}.quat; Data_new{ii}.ft];
        case 1
            Data{ii}  = [Data_new{ii}.X; Data_new{ii}.X_dot; Data_new{ii}.ft];
        case 2
            Data{ii}  = [Data_new{ii}.X; Data_new{ii}.ft];
        case 3
            Data{ii}  = [Data_new{ii}.X; Data_new{ii}.X_dot;];
    end
    Data{ii} = Data{ii}   .* repmat( weights, 1, length(Data{ii}));
end

label_range = unique(True_states{1});

if do_plots
    Data_plot = []; True_states_plot = [];
    ts = [1:length(Data)];
    
    figure('Color',[1 1 1]);
    for i=1:length(ts)
        X = Data{ts(i)};
        Data_plot{i}        = Data{ts(i)}';
        True_states_plot{i} = True_states{ts(i)}';
                                
        % Plot time-series with true labels
        true_states = True_states{ts(i)};
        subplot(length(ts),1,i);
        data_labeled = [X ; true_states];
        plot_title = strcat(dataset_name,{' '}, '[Ground truth for time-series $\mathbf{x}^{(i)}$ i=', num2str(i),']' );
        plotLabeledData( data_labeled, [], plot_title, [], label_range)
    end
    legend1 = legend(data_legends,'Interpreter','LaTex','Fontsize',12);
    set(legend1,...
    'Position',[0.912310956708616 0.235158745421729 0.0781955735963452 0.571190468016125],...
    'Interpreter','latex',...
    'FontSize',12);
               
    % Plot Ground Segmentations
    title_name = {dataset_name,'Ground Truth'};
    title_edge = [0 0 0];
    plot3D_labeledTrajectories(Data_plot, True_states_plot, plot_subsample, title_name, title_edge,  dataset_view);    
end

%%%%%%%%%%%%%%%%%%%%%%  Output variables for algorithms %%%%%%%%%%%%%%%%%%%%%% 
% Data structures for ibp-hmm / icsc-hmm
data = SeqData();
N = length(Data);
for iter = 1:N    
    X = Data{iter};
    labels = True_states{iter};
    data = data.addSeq( X, num2str(iter), labels );
    
    Data{iter} = Data{iter}';
    True_states{iter} = True_states{iter}';
    
end
TruePsi = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%