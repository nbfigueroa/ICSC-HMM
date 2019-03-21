function [h] = plotSingleLabelSegmentation(data, bestPsi, legends, title_name)

% Extract info from 'Best Psi'
K_est = bestPsi.nFeats;
est_states_all = [];
for ii=1:data.N; est_states_all  = [est_states_all bestPsi.Psi.stateSeq(ii).z]; end
label_range = unique(est_states_all);
est_states  = [];

% Plot Segmentation
h = figure('Color',[1 1 1]);
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
    plot_title = strcat(title_name,{' '}, 'for time-series $\mathbf{x}^{(i)}$ i=', num2str(i) );
    plotLabeledData( data_labeled, [], plot_title,  [],label_range)    
end
legend1 = legend(legends,'Interpreter','LaTex','Fontsize',12);
set(legend1,...
    'Position',[0.912310956708616 0.235158745421729 0.0781955735963452 0.571190468016125],...
    'Interpreter','latex',...
    'FontSize',12);


end