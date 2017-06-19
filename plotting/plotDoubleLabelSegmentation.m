function [h] = plotDoubleLabelSegmentation(data, bestPsi, varargin)

% Extract info from 'Best Psi'
K_est = bestPsi.nFeats;

% Generate transform-invariant state sequences
est_states_all = [];
for ii=1:data.N; 
    est_states_all  = [est_states_all bestPsi.Psi.stateSeq(ii).z]; 
end
label_range_z = unique(est_states_all);

if isempty(varargin)
    label_range_s = unique(bestPsi.Psi.Z);
else
    est_labels = varargin{1};
    label_range_s = unique(est_labels);
end


% Plot Segmentation
h = figure('Color',[1 1 1]);
est_states = [];
est_clust_states = [];
est_states_all    = [];
est_clust_states_all    = [];

for i=1:data.N
    
    % Extract data from each time-series    
    X = data.Xdata(:,[data.aggTs(i)+1:data.aggTs(i+1)]);
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states{i}  = bestPsi.Psi.stateSeq(i).z;
    est_states_all  = [est_states_all; est_states{i}'];
     
    % If decoupled model, compute state sequence        
    if isempty(varargin)
        est_clust_states{i} = bestPsi.Psi.stateSeq(i).c;
    else
        clear s c
        s = est_states{i};
        for k=1:length(est_labels)
            c(s==k) = est_labels(k);
        end
        est_clust_states{i} = c;
    end
    est_clust_states_all  = [est_clust_states_all; est_clust_states{i}'];
    
    % Plot Inferred Segments
    subplot(data.N,1,i);
    data_labeled = [X; est_clust_states{i}; est_states{i}];
    plotDoubleLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(i),'), K:',num2str(K_est), ', $K_z$:',num2str(length(label_range_s))), [], label_range_z, label_range_s);    
    
   

end