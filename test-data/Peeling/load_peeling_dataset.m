function [data, TruePsi, Data, True_states, Data_] = load_peeling_dataset( data_path, dim, display, normalize, varargin)

label_range = [1 2 3 4 5];
Data_ = []; 
load(strcat(data_path,'Peeling/proc-data-labeled.mat'))
    
switch dim 
    case 'all'
        dimensions = [1:size(Data{1},1)];
    case 'robots'
        dimensions = [1:size(Data{1},1)-6];
    case 'act+obj'
        dimensions = [1:13 27:size(Data{1},1)];
    case 'active'
        dimensions = [1:13];
end

% Select dimensions
for i=1:length(Data)
    Data{i} = Data{i}(dimensions,:);
end


if display == 1
    ts = [1:length(Data)];
    figure('Color',[1 1 1])
    for i=1:length(ts)
        X = Data{ts(i)};
        true_states = True_states{ts(i)};
        
        % Plot time-series with true labels
        subplot(length(ts),1,i);
        data_labeled = [X ; true_states];
        plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), [], label_range)
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


end



% % Visualize data
% all = 1;
% id = 3;
% if exist('h0','var') && isvalid(h0), delete(h0);end
% h0  = plotPeelingDemo(Data_all, 1, id);
% 
% % Visualize Labeled Data
% id = 3;
% clear X
% X(1:13,:)  = Data_all{id}.active.X;
% X(14:26,:) = [Data_all{id}.passive.X Data_all{id}.passive.X(:,end)];
% 
% x_o = Data_all{id}.object.feats;
% Rate of change of color
% x_o_dot = [zeros(6,1) diff(x_o')'];
% Smoothed out with savitksy golay filter
% x_o_dot = sgolayfilt(x_o_dot', 6, 151)';
% 
% X(27:32,:) = x_o_dot;
% Data{id} = X;
% True_states{id} = ones(1,length(X));