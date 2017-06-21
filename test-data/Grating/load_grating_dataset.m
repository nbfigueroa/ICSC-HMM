function [data, TruePsi, Data, True_states, Data_] = load_grating_dataset( data_path, type, display, full, varargin)


label_range = [1 2 3];

% Data structures for hmm / hdp-hmm

        
if full == 1 % Load the 12 time-series
    switch type
        case 'robot'
            load(strcat(data_path,'/Grating/CarrotGrating_robot.mat'))
            Data_ = Data;
        case 'grater'
            load(strcat(data_path,'/Grating/CarrotGrating_grater.mat'))
            Data_ = Data;
        case 'mixed'
            load(strcat(data_path,'/Grating/CarrotGrating_robot.mat'))
            Data_ = Data;
            load(strcat(data_path,'/Grating/CarrotGrating_grater.mat'))
            for i=2:2:length(Data)
                Data{i} = Data_{i};
            end
            clear Data_
            Data_ = Data;
    end

    if display == 1
        ts = [1:4];
        figure('Color',[1 1 1])
        for i=1:length(ts)
            X = Data{ts(i)};
            true_states = True_states{ts(i)};
            
            % Plot time-series with true labels
            subplot(length(ts),1,i);
            data_labeled = [X true_states]';
            plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), [], label_range)
        end
        
        
        figure('Color',[1 1 1])
        ts = [5:8];
        for i=1:length(ts)
            X = Data{ts(i)};
            true_states = True_states{ts(i)};
            
            % Plot time-series with true labels
            subplot(length(ts),1,i);
            data_labeled = [X true_states]';
            plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), [], label_range)
        end
        
        figure('Color',[1 1 1])
        ts = [9:12];
        for i=1:length(ts)
            X = Data{ts(i)};
            true_states = True_states{ts(i)};
            
            % Plot time-series with true labels
            subplot(length(ts),1,i);
            data_labeled = [X true_states]';
            plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), [], label_range)
        end
    end
else % Load the 6 time-series
    
    switch type
        case 'robot'
            load(strcat(data_path,'/Grating/CarrotGrating_robot.mat'))
            Data_ = Data; True_states_ = True_states;
            clear Data True_states
            iter = 1;
            for i=1:2:length(Data_)
                Data{iter} = Data_{i};
                True_states{iter} = True_states_{i};
                iter = iter + 1;
            end
        case 'grater'
            load(strcat(data_path,'/Grating/CarrotGrating_grater.mat'))
            Data_ = Data; True_states_ = True_states;
            clear Data True_states
            iter = 1;
            for i=1:2:length(Data_)
                Data{iter} = Data_{i};
                True_states{iter} = True_states_{i};
                iter = iter + 1;
            end
        case 'mixed'
            load(strcat(data_path,'/Grating/CarrotGrating_robot.mat'))
            Data_ = Data;
            load(strcat(data_path,'/Grating/CarrotGrating_grater.mat'))
            for i=2:2:length(Data)
                Data{i} = Data_{i};
            end
            
            Data_ = Data; True_states_ = True_states;
            clear Data True_states
            iter = 1;
            for i=1:6
                Data{iter} = Data_{i};
                True_states{iter} = True_states_{i};
                iter = iter + 1;
            end
    end
    Data_ = Data;
    
    
    % Convert positions to velocities
    if ~isempty(varargin)
        if varargin{1}==1
            for i=1:length(Data)
                clear X3d
                X3d(1:3,:) = Data{i}(:,1:3)';
                X3d_dot = [zeros(3,1) diff(X3d')'];
                % Smoothed out with savitksy golay filter
                X3d_dot = 100*sgolayfilt(X3d_dot', 3, 151)';
                Data{i}(:,1:3) = X3d_dot';
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
            data_labeled = [X true_states]';
            plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), [], label_range)
        end
    end
    
end

% Data structures for ibp-hmm / icsc-hmm
data = SeqData();
N = length(Data);
for iter = 1:N    
    X = Data{iter}';
    labels = True_states{iter}';
    data = data.addSeq( X, num2str(iter), labels );
end

TruePsi = [];



end


