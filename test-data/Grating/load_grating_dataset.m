function [data, TruePsi, Data, True_states] = load_grating_dataset( data_path, type, display, full)


label_range = [1 2 3];

% Data structures for hmm / hdp-hmm
switch type
    case 'same'
        
        if full == 1 % Load the 12 time-series
            load(strcat(data_path,'/Grating/CarrotGrating.mat'))
            
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
            load(strcat(data_path,'/Grating/CarrotGrating.mat'))
            Data_ = Data; True_states_ = True_states;
            clear Data True_states
            iter = 1;
            for i=1:2:length(Data_)
                Data{iter} = Data_{i};
                True_states{iter} = True_states_{i};
                iter = iter + 1;
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
        
    case 'transformed'
        
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


