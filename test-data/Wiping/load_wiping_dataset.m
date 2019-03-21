function [data, TruePsi, Data, True_states, data_legends, dataset_name, plot_subsample, dataset_view] = load_wiping_dataset(data_path, dataset_type, do_plots, variables)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Load One of the Wiping Datasets    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
do_plot_ind = 0;

%%%%%%%%%%%%%%%%%%%%  Input variables for loading function %%%%%%%%%%%%%%%%%%%%%%  
% % Select Dataset to Load
% dataset_type = 2; % 1: Rim Cover
%                   % 2: Door Fender  
% 
% data_path  = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';
% do_plots   = 1;
% do_plot_ind = 0;
%                  
% % Select which Variables to use
% variables = 1;
% % Rotate force/torque measurements to global reference frame
 rotate_ft = 0;        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch dataset_type
    case 1
        load(strcat(data_path,'Wiping/demos_rim_processed_data.mat'))
        dataset_name   = 'Task 2: Rim Wiping';
        plot_subsample = 2; 
        dataset_view = [-80 16];
        
        
        True_states{1} = [ones(1,234) 2*ones(1,547) 3*ones(1,181)];
        True_states{2} = [ones(1,208) 2*ones(1,794) 3*ones(1,237)];    
        True_states{3} = [ones(1,243) 2*ones(1,623) 3*ones(1,230)];    
        True_states{4} = [ones(1,284) 2*ones(1,940) 3*ones(1,233)];    
        True_states{5} = [ones(1,304) 2*ones(1,610) 3*ones(1,206)];    
        True_states{6} = [ones(1,224) 2*ones(1,831) 3*ones(1,176) ...
                          ones(1,174) 2*ones(1,765) 3*ones(1,214) ...
                          ones(1,230) 2*ones(1,651) 4*ones(1,190) 2*ones(1,595) 3*ones(1,171)];    
        label_range = [1:3];
          
        % Replace force/torque signals in 5-th series
        ft_tmp = proc_data{3}.X(8:13,250:end);        
        end_ = length(ft_tmp)+ 303 - length(proc_data{5}.X);
        proc_data{5}.X(8:13,1:244)  = proc_data{3}.X(8:13,1:244); 
        proc_data{5}.X(8:13,245:303)  = proc_data{3}.X(8:13,186:244); 
        proc_data{5}.X(8:13,304:end) = ft_tmp(:,1:end-end_); 
        
        % Shift 3rd/4-th/5-th time-series
        proc_data{3}.X(2,:)    = proc_data{3}.X(2,:) + 0.05 ;
        
        proc_data{4}.X(1,:)    = proc_data{4}.X(1,:) - 0.05 ;
        proc_data{4}.X(2,:)    = proc_data{4}.X(2,:) - 0.075 ;
        proc_data{4}.X(3,:)    = proc_data{4}.X(3,:) + 0.015 ;
        
        proc_data{5}.X(1,:)    = proc_data{5}.X(1,:) - 0.15;
        proc_data{5}.X(2,:)    = proc_data{5}.X(2,:) + 0.05;
        proc_data{5}.X(3,:)    = proc_data{5}.X(3,:) + 0.025;
                
        % Remove an extra measurement on the 6-th time-series
        proc_data{6}.X         = proc_data{6}.X(:,1:1238);
        proc_data{6}.X_dot     = proc_data{6}.X_dot(:,1:1238);
        proc_data{6}.H         = proc_data{6}.H(:,:,1:1238);
        proc_data{6}.J         = proc_data{6}.J(:,1:1238);
        True_states{6}         = True_states{6}(1,1:1238);
        
        % Remove final time-series
        proc_data(1:2) = [];
        True_states(1:2) = [];
        
        % Switch time-series order
        proc_data_ = proc_data;
        True_states_ = True_states;
        
        proc_data{1} = proc_data_{4};
        True_states{1} = True_states_{4};
        proc_data{2} = proc_data_{1};
        True_states{2} = True_states_{1};
        proc_data{3} = proc_data_{2};
        True_states{3} = True_states_{2};
        proc_data{4} = proc_data_{3};
        True_states{4} = True_states_{3};
        
        % Final fixes
        proc_data{3}.X(10,1:322)  = zeros(1,322); 

        
        switch variables
            case 0 
                weights = [ones(3,1); 1*ones(3,1); 2*ones(3,1); 1/30*ones(3,1); 0.75*ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', '$\alpha$','$\beta$','$\gamma$', ...
                                '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$', ...
                                '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};
            
            case 1
                weights = [ones(3,1); ones(3,1); 1/25*ones(3,1); 1/2*ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', ...
                                '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$', ...
                                '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};
            case 2
                weights = [ones(3,1); 1/25*ones(3,1); 1/2*ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};
            case 3
                weights = [ones(3,1); ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$'};
        end
        
    case 2
        load(strcat(data_path,'Wiping/demos_fender_processed_data.mat'))        
        dataset_name = 'Task 1: Fender Wiping';
        dataset_view = [160 16];
        plot_subsample = 3; 
        
        True_states{1} = 3*ones(1,length(proc_data{1}.X));               
        True_states{2} = [ones(1,282) 2*ones(1,257)  ones(1,296) ...
                                      2*ones(1,290)  ones(1,293) ...
                                      2*ones(1,291)  ones(1,352) ...
                                      2*ones(1,264)  ones(1,213)];
        
        % Remove an extra measurement on the 3-rd time-series
        proc_data{3}.X         = proc_data{3}.X(:,1:2315);
        proc_data{3}.X_dot     = proc_data{3}.X_dot(:,1:2315);
        proc_data{3}.H         = proc_data{3}.H(:,:,1:2315);
        proc_data{3}.J         = proc_data{3}.J(:,1:2315);
        
        % Replace Force/Torque signals in end series
        proc_data{2}.X(8:13,1:69) = zeros(6,69);
        proc_data{2}.X(8:13,70:length(proc_data{3}.X)+69) = proc_data{3}.X(8:13,:);
        tmp = proc_data{2}.X(8:13,634:end);
        proc_data{2}.X(8:13,634:end) = [zeros(6,30) tmp(:,1:end-30)];
        proc_data{2}.X(8:13,1217:end) = [proc_data{2}.X(8:13,70:1217) zeros(6,174)] ;
        tmp = proc_data{2}.X(8:13,1806:end);
        proc_data{2}.X(8:13,1806:end) = [zeros(6,100) tmp(:,1:end-100)];
        
        True_states{3} = [ones(1,212) 2*ones(1,256)  ones(1,259) ...
                                      2*ones(1,270)  ones(1,228) ...
                                      2*ones(1,401)  ones(1,227) ...
                                      2*ones(1,279)  ones(1,183)];

        % Shift 2nd time-series
        % With only this change it worked 
        proc_data{2}.X(1,:)    = proc_data{2}.X(1,:) - 0.25 ;

        % Remove an extra measurement on the 2-nd time-series
        proc_data{2}.X         = proc_data{2}.X(:,150:end-50);
        proc_data{2}.X_dot     = proc_data{2}.X_dot(:,150:end-50);
        proc_data{2}.H         = proc_data{2}.H(:,:,150:end-50);
        proc_data{2}.J         = proc_data{2}.J(:,150:end-50);
        True_states{2}         = True_states{2}(150:end-50);
        proc_data{3}.X         = proc_data{3}.X(:,50:end-10);
        proc_data{3}.X_dot     = proc_data{3}.X_dot(:,50:end-10);
        proc_data{3}.H         = proc_data{3}.H(:,:,50:end-10);
        proc_data{3}.J         = proc_data{3}.J(:,50:end-10);
        True_states{3}         = True_states{3}(50:end-10);
        
        % Remove first time-series
        proc_data(1) = [];
        True_states(1) = [];
        
        label_range = [1:2];
        

        switch variables
            case 0 
                weights = [ones(3,1); 1/2*ones(3,1); 2*ones(3,1); 1/50*ones(3,1); 1/5*ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', '$\alpha$','$\beta$','$\gamma$', ...
                                '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$', ...
                                '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};
                
            case 1
                weights = [ones(3,1); 2*ones(3,1); 1/50*ones(3,1); 1/5*ones(3,1)];                
                data_legends = {'$x_1$','$x_2$','$x_3$', ...
                                '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$', ...
                                '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};

            case 2
                weights = [ones(3,1); 1/25*ones(3,1); 1/5*ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', '$f_x$','$f_y$','$f_z$','$\tau_x$','$\tau_y$','$\tau_z$'};
                
            case 3
                weights = [ones(3,1); ones(3,1)];
                data_legends = {'$x_1$','$x_2$','$x_3$', '$\dot{x}_1$','$\dot{x}_2$','$\dot{x}_3$'};
        end      
end

% Create Euler Angle representation of Rotations
for p=1:length(proc_data)
    H_tmp = proc_data{p}.H;
    xyz = zeros(3,size(H_tmp,3));
    for i=1:length(xyz)
        xyz(:,i) = R2rpy(H_tmp(1:3,1:3,i));
    end
    proc_data{p}.euler = xyz;
end


if rotate_ft
    for p=1:length(proc_data)
        ft_tmp = proc_data{p}.X(8:end,:);
        H_tmp  = proc_data{p}.H;
        for i=1:length(ft_tmp)
            R = eye(6);
            R(1:3,1:3) = H_tmp(1:3,1:3,i);
            R(4:6,4:6) = H_tmp(1:3,1:3,i);
            ft_tmp(:,i) = (R')*ft_tmp(:,i);
        end
        proc_data{p}.X(8:end,:) = ft_tmp;
    end    
end

            
N = length(proc_data);
Data       = [];
for ii=1:N
    switch variables
        case 0
            Data{ii}  = [proc_data{ii}.X(1:3,:); proc_data{ii}.X_dot(4:6,:); proc_data{ii}.X_dot(1:3,:); proc_data{ii}.X(8:end,:)];
        case 1
            Data{ii}  = [proc_data{ii}.X(1:3,:); proc_data{ii}.X_dot(1:3,:); proc_data{ii}.X(8:end,:)];
        case 2
            Data{ii}  = [proc_data{ii}.X(1:3,:); proc_data{ii}.X(8:end,:)];
        case 3
            Data{ii}  = [proc_data{ii}.X(1:3,:); proc_data{ii}.X_dot(1:3,:)];
    end
    Data{ii} = Data{ii}   .* repmat( weights, 1, length(Data{ii}));
    
    if do_plot_ind
        title_name = strcat(dataset_name, {' '}, 'Time-Series', {' '}, num2str(ii));
        plotEEDatav2( Data{ii},title_name ) 
    end
end

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