function [h] = plotDoubleLabelSegmentation2(Data, est_states, est_super_states, legends, title_name, label_range, est_labels, varargin)

if nargin == 8
    flipit = 1;
else
    flipit = 0;
end

ts = [1:length(Data)];
h= figure('Color',[1 1 1]);
for i=1:length(ts)
    if flipit
        X = Data{ts(i)};
        est_states_i       = est_states{i};
        est_super_states_i = est_super_states{i};

    else
        X = Data{ts(i)};
        est_states_i       = est_states{i};
        est_super_states_i = est_super_states{i}';
    end
    size(X)
    size(est_states_i)
    size(est_super_states_i)
    
    % Plot time-series with estimated labels and estimated super labels
    subplot(length(ts),1,i);
    data_labeled = [X est_super_states_i est_states_i]';
    size(data_labeled)
    label_range_s = label_range
    label_range_z = unique(est_labels)
    
    plot_title = strcat(title_name,{' '}, 'for time-series $\mathbf{x}^{(i)}$ i=', num2str(i) );    
    plotDoubleLabeledData( data_labeled, [], plot_title, [], label_range_z, label_range_s);    
end


legend1 = legend(legends,'Interpreter','LaTex','Fontsize',12);
set(legend1,...
    'Position',[0.912310956708616 0.235158745421729 0.0781955735963452 0.571190468016125],...
    'Interpreter','latex',...
    'FontSize',12);


end