function [data, TruePsi, Data, True_states, Data_o] = load_peeling_dataset( data_path, dim, display, normalize, varargin)

label_range = [1 2 3 4 5];
load(strcat(data_path,'Peeling/proc-data-labeled.mat'))

load(strcat(data_path,'Peeling/proc-labels.mat'))
switch dim 
    case 'all'
        dimensions = [1:size(Data{1},1)];                     
        
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
        dimensions = [1:size(Data{1},1)-6];
    case 'act+obj'
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
        
    case 'active'
        load(strcat(data_path,'Peeling/proc-data-noObj.mat'))
        Data{4} = Data_noObj{1}; True_states{4} = True_states_noObj{1};
        Data{5} = Data_noObj{2}; True_states{5} = True_states_noObj{2};
        dimensions = [1:13];        
end
Data_o = Data;   

% Select dimensions
for i=1:length(Data)
    Data{i} = Data{i}(dimensions,:);
end


% Convert positions to velocities
if ~isempty(varargin)
    if varargin{2}==1
        for i=1:length(Data)
            clear X3d
            X3d(1:3,:) = Data{i}(1:3,:);
            X3d_dot = [zeros(3,1) diff(X3d')'];
            % Smoothed out with savitksy golay filter
            X3d_dot = 100*sgolayfilt(X3d_dot', 3, 151)';
            Data{i}(1:3,:)       = X3d_dot;

            if strcmp(dim,'all') || strcmp(dim,'robots')
                clear X7d
                X7d(1:7,:) = Data{i}(14:20,:);
                X7d_dot = [zeros(7,1) diff(X7d')'];
                
                % Smoothed out with savitksy golay filter
                X7d_dot        = 100*sgolayfilt(X7d_dot', 7, 151)';                                
                Data{i}(14:20,:) = X7d_dot;
            end

        end
    end
end

if normalize > 0
    
    if isempty(varargin)
        X = Data{1};
        weights = ones(1,size(X,1))';
    else
        weights = varargin{1};
    end
    
    
    for i=1:length(Data)
        X = Data{i};
        
        if normalize == 1
            mean_X     = mean(X,2);
            X_zeroMean = X - repmat( mean_X, 1, length(X));
            Data{i} = X_zeroMean;
        else
            X_weighted = X   .* repmat( weights, 1, length(X));
            mean_X     = mean(X_weighted,2);
            X_zeroMean = X_weighted - repmat( mean_X, 1, length(X));
            Data{i} = X_zeroMean;
        end
    end
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
    Data_o{iter} = Data_o{iter}';
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