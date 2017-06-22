function [results] = computeSegmClustmetrics(true_states_all, Best_Psi, varargin)

% Number of runs
T = length(Best_Psi);
N = length(Best_Psi(1).Psi.stateSeq)

% Segmentation Metric Arrays
hamming_distance     = zeros(1,T);
global_consistency   = zeros(1,T);
variation_info       = zeros(1,T);
inferred_states      = zeros(1,T);
inferred_state_clust = zeros(1,T);

nFeats      = zeros(1,T);
nClusts = zeros(1,T);

% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);

if ~isempty(varargin)
    est_labels = varargin{1};
end


for run=1:T    
    clear Psi
    est_states_all = [];
    est_clusts_all = [];
    
    % Extract Estimated States for all sequences
    Psi = Best_Psi(run).Psi;
    for j=1:N
        est_states_all = [est_states_all Best_Psi(run).Psi.stateSeq(j).z];
        
        if isempty(varargin)
            if isfield(Best_Psi(run).Psi.stateSeq(j), 'c')
                est_clusts_all = [est_clusts_all Best_Psi(run).Psi.stateSeq(j).c];
            end
        else
            clear s c
            s = Best_Psi(run).Psi.stateSeq(j).z;
            if iscell(est_labels)
                labels = est_labels{run};
            else
                labels = est_labels;
            end
            for k=1:length(labels)
                c(s==k) = labels(k);
            end
            est_clusts_all = [est_clusts_all c];
        end
        
    end
    
     % Segmentation Metrics per run considering transform-dependent state
     % sequence given by F         
    [relabeled_est_states_all, hamming_distance(run),~,~] = mapSequence2Truth(true_states_all,est_states_all);
    [~,global_consistency(run), variation_info(run)] = compare_segmentations(true_states_all,est_states_all);
    inferred_states(run)      = length(unique(est_states_all));
    nFeats(run)               = Best_Psi(run).nFeats;
    
    % Cluster Metrics per run considering transform-invariant state
    % sequences given by Z(F)
    
    if isempty(varargin)
        if isfield(Best_Psi(run).Psi.stateSeq(j), 'c')
            [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, est_clusts_all);
            inferred_state_clust(run) = length(unique(est_clusts_all));
            nClusts(run)               = Best_Psi(run).nClusts;
        else
            [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, relabeled_est_states_all);
        end
    else
        [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, est_clusts_all);
        inferred_state_clust(run) = length(unique(est_clusts_all));
    end
end

inferred_states
nFeats

inferred_state_clust
nClusts
    
% Make struct with metrics
results = struct();
results.hamming_distance     = hamming_distance;
results.global_consistency   = global_consistency;
results.variation_info       = variation_info;
results.inferred_states      = inferred_states;
results.cluster_purity       = cluster_purity;
results.cluster_NMI          = cluster_NMI;
results.cluster_F            = cluster_F;

if isempty(varargin)
    if isfield(Best_Psi(run).Psi.stateSeq(j), 'c')
        results.inferred_state_clust = inferred_state_clust;
        % Overall Stats for HMM segmentation and state clustering
        fprintf('*** ICSC-HMM Results*** \n Optimal Transform-Dependent States: %3.3f (%3.3f) \n Optimal Transform-Invariant States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) ...
            mean(inferred_state_clust) std(inferred_state_clust) mean(hamming_distance) std(hamming_distance) mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])
    else
        % Overall Stats for HMM segmentation and state clustering
        fprintf('*** IBP-HMM Results*** \n Optimal Feature States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) ...
            mean(hamming_distance) std(hamming_distance) mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])
    end
else
    results.inferred_state_clust = inferred_state_clust;
    % Overall Stats for HMM segmentation and state clustering
    fprintf('*** IBP-HMM + SPCM-CRP Results*** \n Optimal Transform-Dependent States: %3.3f (%3.3f) \n Optimal Transform-Invariant States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) ...
            mean(inferred_state_clust) std(inferred_state_clust) mean(hamming_distance) std(hamming_distance) mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])
end

end