function [data, TruePsi, Data, True_states] = load_rolling_dataset( data_path, display)


label_range = [1 2 3];
        
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


