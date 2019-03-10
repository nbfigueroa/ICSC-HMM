function [h] = plotSingleLabelSegmentation(data, bestPsi)

% Extract info from 'Best Psi'
K_est = bestPsi.nFeats;
est_states_all = [];
for ii=1:data.N; est_states_all  = [est_states_all bestPsi.Psi.stateSeq(ii).z]; end
label_range = unique(est_states_all)
est_states = [];

% Plot Segmentation
h = figure('Color',[1 1 1]);
true_states_all   = [];
est_states_all    = [];

for i=1:data.N
    
    % Extract data from each time-series    
    X = data.Xdata(:,[data.aggTs(i)+1:data.aggTs(i+1)]);
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states{i}  = bestPsi.Psi.stateSeq(i).z;    
    est_states_all  = [est_states_all; est_states{i}'];
    
    % Plot Inferred Segments
    subplot(data.N,1,i);
    data_labeled = [X; est_states{i}];
    plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(i),'), K:',num2str(K_est)),  {'x_1','x_2'},label_range)
    
end



end