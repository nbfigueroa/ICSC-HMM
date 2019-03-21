function [handle, est_states_all, true_states_all, est_states, est_states_plot] = plotHDPLabelSegmentation(data_struct,BestChain,label_range, label_range_mapped, TruePsi, True_states, title_name, data_legends)
handle = figure('Color',[1 1 1]);
true_states_all   = [];
est_states_all    = [];

est_states = [];
est_states_plot = [];

for i=1:length(data_struct)
    X = data_struct(i).obs;
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states_raw  = BestChain.stateSeq(i).z;
    est_states_mapped = zeros(1,length(est_states_raw));
    for j=1:length(est_states_raw)
        est_states_mapped(j) = find(est_states_raw(j) == label_range);
    end
    est_states{i} = est_states_mapped;
    est_states_plot{i} = est_states{i}';
    
    % Stack labels for state clustering metrics
    if isfield(TruePsi, 'sTrueAll')       
        true_states = TruePsi.s{i}';        
    else
        true_states = True_states{i};        
    end
    true_states_all = [true_states_all; true_states];
    est_states_all  = [est_states_all; est_states{i}'];
    
    % Plot Inferred Segments
    subplot(length(data_struct),1,i);
    data_labeled = [X; est_states{i}];
    plot_title = strcat(title_name,{' '}, 'for time-series $\mathbf{x}^{(i)}$ i=', num2str(i) );
    plotLabeledData( data_labeled, [], plot_title,  [], label_range_mapped) 
end
legend1 = legend(data_legends,'Interpreter','LaTex','Fontsize',12);
set(legend1,...
    'Position',[0.912310956708616 0.235158745421729 0.0781955735963452 0.571190468016125],...
    'Interpreter','latex',...
    'FontSize',12);

end