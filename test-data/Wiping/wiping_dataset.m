%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Load One of the Wiping Datasets    %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
dataset_type = 1; % 1: Rim Cover
                  % 2: Door Fender  
data_path = '/home/nbfigueroa/Dropbox/PhD_papers/journal-draft/new-code/ICSC-HMM/test-data/';
do_plot = 0;

switch dataset_type
    case 1
        load(strcat(data_path,'Wiping/demos_rim_processed_data.mat'))
        dataset_name = 'Rim-Wiping';
        
    case 2
        load(strcat(data_path,'Wiping/demos_fender_processed_data.mat'))
        dataset_name = 'Fender-Wiping';
end

weights = [ones(3,1); 5*ones(3,1); 1/25*ones(3,1); 1/5*ones(3,1)];

N = length(proc_data);
Data       = [];
True_states = [];
for ii=1:N
    if ii == 5
        Data{ii}        = [proc_data{ii}.X(1:3,:); proc_data{ii}.X_dot(4:6,:); proc_data{2}.X(8:end,1:length(proc_data{ii}.X))];
    else
        Data{ii}        = [proc_data{ii}.X(1:3,:); proc_data{ii}.X_dot(4:6,:); proc_data{ii}.X(8:end,:)];
    end
    
    Data{ii} = Data{ii}   .* repmat( weights, 1, length(Data{ii}));    
    True_states{ii} = ii*ones(1,length(Data{ii}));
    if do_plot
        title_name = strcat(dataset_name, {' '}, 'Time-Series', {' '}, num2str(ii));
        plotEEDatav2( Data{ii},title_name )
    end
end

%
if true
    ts = [1:length(Data)];
    figure('Color',[1 1 1])
    for i=1:length(ts)
        X = Data{ts(i)};
        true_states = True_states{ts(i)};
        
        % Plot time-series with true labels
        subplot(length(ts),1,i);
        data_labeled = [X ; true_states];
        plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), [], [])
    end
end

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