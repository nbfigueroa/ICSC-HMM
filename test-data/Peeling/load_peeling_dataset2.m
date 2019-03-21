function [data, TruePsi, Data, True_states, data_legends, dataset_name, plot_subsample, dataset_view] = load_peeling_dataset2(data_path, do_plots, dim, time_series)

% Load Datasets and Ground Truth Labels
label_range = [1 2 3 4 5];
load(strcat(data_path,'Peeling/proc-data-labeled.mat'))
load(strcat(data_path,'Peeling/proc-labels.mat'))
dataset_name = 'Task 4: Peeling';
plot_subsample = 5; 

if strcmp(dim,'robots')
    dataset_view = [-195 13];
else    
    dataset_view = [-145 7];
end

switch dim 
    case 'all'
        dimensions = [1:size(Data{1},1)];                             
        weights = [ones(1,3) ones(1,4) 1/10*ones(1,6) ones(1,3) ...
            2*ones(1,4) 1/20*ones(1,6) 2*ones(1,6) ]'; % all        
        
        for i=1:length(Data)
            clear X6f
            X6f(1:6,:) = Data{i}(end-5:end,:);
            
            % Smooth out color features
            sf = 200;
            X6f_s = [smooth(X6f(1,:),sf)'; smooth(X6f(2,:),sf)'; smooth(X6f(3,:),sf)'; ...
                     smooth(X6f(4,:),sf)'; smooth(X6f(5,:),sf)'; smooth(X6f(6,:),sf)'];           
            Data{i}(end-5:end,:) = X6f_s;
        end
                
    case 'robots'
        load(strcat(data_path,'Peeling/proc-data-noObj.mat'))
        Data{4} = Data_noObj{1}; True_states{4} = True_states_noObj{1};
        Data{5} = Data_noObj{2}; True_states{5} = True_states_noObj{2};
%         dimensions = [1:size(Data{1},1)-6];             
%         weights = [1*ones(1,3) 1/2*ones(1,4) 1/10*ones(1,3)  1/5*ones(1,3) ...
%                    1*ones(1,3) 1/2*ones(1,4) 1/100*ones(1,3)  1/50*ones(1,3)]';                
%         data_legends = {'$x_1^a$','$x_2^a$','$x_3^a$', ...
%             '$q_i^a$','$q_j^a$','$q_k^a$','$q_w^a$', ...
%             '$f_x^a$','$f_y^a$','$f_z^a$','$\tau_x^a$','$\tau_y^a$','$\tau_z^a$', ...
%             '$x_1^p$','$x_2^p$','$x_3^p$', ...
%             '$q_i^p$','$q_j^p$','$q_k^p$','$q_w^p$',...
%             '$f_x^p$','$f_y^a$','$f_z^p$','$\tau_x^p$','$\tau_y^p$','$\tau_z^p$'};   
        
        dimensions = [1:size(Data{1},1)-16];             
        weights = [1*ones(1,3) 1/2*ones(1,4) 1/10*ones(1,3)  1/5*ones(1,3) ...
                   1*ones(1,3)]';                               
        data_legends = {'$x_1^a$','$x_2^a$','$x_3^a$', ...
            '$q_i^a$','$q_j^a$','$q_k^a$','$q_w^a$', ...
            '$f_x^a$','$f_y^a$','$f_z^a$','$\tau_x^a$','$\tau_y^a$','$\tau_z^a$', ...
            '$x_1^p$','$x_2^p$','$x_3^p$'};   
        
    case 'act+obj'
        weights = [ones(1,7) 1/10*ones(1,6) 2*ones(1,6)]'; % act+obj
        dimensions = [1:13 27:size(Data{1},1)];
        
        for i=1:length(Data)
            clear X6f
            X6f(1:6,:) = Data{i}(end-5:end,:);
            % Smooth out color features
            sf = 200;
            X6f_s = [smooth(X6f(1,:),sf)'; smooth(X6f(2,:),sf)'; smooth(X6f(3,:),sf)'; ...
                     smooth(X6f(4,:),sf)'; smooth(X6f(5,:),sf)'; smooth(X6f(6,:),sf)'];           
            Data{i}(end-5:end,:) = X6f_s;
        end
        data_legends = {'$x_1^a$','$x_2^a$','$x_3^a$', ...
            '$\dot{x}_1^a$','$\dot{x}_2^a$','$\dot{x}_3^a$', ...
            '$q_i^a$','$q_j^a$','$q_k^a$','$q_w^a$', ...
            '$f_x^a$','$f_y^a$','$f_z^a$','$\tau_x^a$','$\tau_y^a$','$\tau_z^a$'};   
        
    case 'active'
        load(strcat(data_path,'Peeling/proc-data-noObj.mat'))
        Data{4} = Data_noObj{1}; True_states{4} = True_states_noObj{1};
        Data{5} = Data_noObj{2}; True_states{5} = True_states_noObj{2};
        dimensions = [1:13];        
        weights = [1*ones(1,3) 1*ones(1,4) 1/10*ones(1,3)  1/5*ones(1,3)  2*ones(1,3) ]'; % active    
        data_legends = {'$x_1^a$','$x_2^a$','$x_3^a$', ...            
            '$q_i^a$','$q_j^a$','$q_k^a$','$q_w^a$', ...
            '$f_x^a$','$f_y^a$','$f_z^a$','$\tau_x^a$','$\tau_y^a$','$\tau_z^a$', ...
            '$\dot{x}_1^a$','$\dot{x}_2^a$','$\dot{x}_3^a$'};   
end


% Select Specific Time Series
Data(time_series+1:end) = [];
True_states(time_series+1:end) = [];

% Select dimensions and fix/add variables
dt = 1/100; window_size = 13; 
crop_size = (window_size+1)/2;
for i=1:length(Data)
    Data{i} = Data{i}(dimensions,:);
    
    % Adjust force bias on first 3 demonstrations
    if i < 4
        Data{i}(10,:) = Data{i}(10,:) + 10 ;
        fz_tmp = -Data{i}(8,:) - 5;
        fx_tmp = -Data{i}(10,:) - 5;
    else
        fz_tmp = -Data{i}(8,:);
        fx_tmp = -Data{i}(10,:);
    end
    % Adjust force directions    
    Data{i}(8,:)  = fx_tmp ;
    Data{i}(10,:) = fz_tmp ; 
    
    % Adjust force bias on first 3 demonstrations
%     if strcmp(dim,'robots') && i < 4
%         Data{i}(22,:) = Data{i}(22,:) + 10 ; 
%         Data{i}(23,:) = Data{i}(23,:) - 20 ; 
%     end         
    
    % Create velocity vectors and add/crop time-series
    if strcmp(dim,'active')
        dx_nth    = sgolay_time_derivatives(Data{i}(1:3,:)', dt, 2, 3, window_size);
        X3d       = dx_nth(:,:,1)';
        X3d_dot   = dx_nth(:,:,2)';
        quat_tmp  = Data{i}(4:7,crop_size:end-crop_size);
        ft_tmp    = Data{i}(8:end,crop_size:end-crop_size);
        X_tmp     = [X3d; quat_tmp; ft_tmp; X3d_dot];
        size(X_tmp)
        Data{i}   = X_tmp;      
        True_states{i} = True_states{i}(1,crop_size:end-crop_size);
    end              
        
end

% Processing after loading
Data_new = Data;   

N = length(Data_new);
Data   = [];
for ii=1:N
%     switch variables
%         case 0
%             Data{ii}  = [Data_new{ii}.X; Data_new{ii}.X_dot; Data_new{ii}.quat; Data_new{ii}.ft];
%         case 1
%             Data{ii}  = [Data_new{ii}.X; Data_new{ii}.X_dot; Data_new{ii}.ft];
%         case 2
%             Data{ii}  = [Data_new{ii}.X; Data_new{ii}.ft];
%         case 3
%             Data{ii}  = [Data_new{ii}.X; Data_new{ii}.X_dot;];
%     end
    Data{ii} = Data_new{ii};
    size(Data{ii})
    Data{ii} = Data{ii}   .* repmat( weights, 1, length(Data{ii}));
end

label_range = unique(True_states{1});

if do_plots
    Data_plot = []; True_states_plot = [];
    ts = [1:length(Data)];
    
    figure('Color',[1 1 1]);
    for i=1:length(ts)
        X = Data{ts(i)};
        
        if strcmp(dim,'robots')
            Data_plot{i}        = [Data{ts(i)}(1:3,:)  Data{ts(i)}(14:16,:)]' ;
            True_states_plot{i} = [True_states{ts(i)} True_states{ts(i)}]';
        else            
            Data_plot{i}        = Data{ts(i)}';
            True_states_plot{i} = True_states{ts(i)}';
        end
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
axis equal

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


end

