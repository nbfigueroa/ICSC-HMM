%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo scripts for the ICSC-HMM Segmentation Algorithm proposed in:
%
% N. Figueroa and A. Billard, “Transform-Invariant Clustering of SPD Matrices 
% and its Application on Joint Segmentation and Action Discovery}”
% Arxiv, 2017. 
%
% Author: Nadia Figueroa, PhD Student., Robotics
% Learning Algorithms and Systems Lab, EPFL (Switzerland)
% Email address: nadia.figueroafernandez@epfl.ch  
% Website: http://lasa.epfl.ch
% November 2016; Last revision: 25-May-2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                    --Select a Dataset to Test--                       %%    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1) Toy 2D dataset, 3 Unique Emission models, 3 time-series, same swicthing
clc; clear all; close all;
N_TS = 3; display = 2 ; % 0: no-display, 1: raw data in one plot, 2: ts w/labels
[~, Data, True_states] = genToyHMMData_Gaussian( N_TS, display ); 

%% 2) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[~, ~, Data, True_states] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Run E-M Model Selection for HMM with 10 runs in a range of K     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model Selection for HMM
K_range = [1:10]; repeats = 10; 
hmm_eval(Data, K_range, repeats)

%%  Fit HMM with 'optimal' K and Apply Viterbi for Segmentation
% Set "Optimal " GMM Hyper-parameters
K = 4; T = 10;
ts = [1:length(Data)];

% Segmentation Metric Arrays
hamming_distance   = zeros(1,T);
global_consistency = zeros(1,T);
variation_info     = zeros(1,T);
% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);
% Model Metric Arrays
logliks        = zeros(length(ts),T);
label_range    = [1:K];

for run=1:T
    % Fit HMM with 'optimal' K to set of time-series
    [p_start, A, phi, loglik] = ChmmGauss(Data, K);
        
    true_states_all   = [];
    est_states_all    = [];
    
    % Calculate p(X) & viterbi decode fo reach time-series
    if run==T
        if exist('h0','var') && isvalid(h0), delete(h0);end
        h0 = figure('Color',[1 1 1]);
    end
    
    for i=1:length(ts)
        X = Data{ts(i)};
        true_states = True_states{ts(i)};
        logp_xn_given_zn = Gauss_logp_xn_given_zn(Data{i}, phi);
        [~,~, logliks(i,run)] = LogForwardBackward(logp_xn_given_zn, p_start, A);
        est_states = LogViterbiDecode(logp_xn_given_zn, p_start, A);                       
        
        % Stack labels for state clustering metrics        
        true_states_all = [true_states_all; true_states];
        est_states_all  = [est_states_all; est_states];
        
        % Plot segmentation Results on each time-series
        if run==T
            subplot(length(ts),1,i);
            data_labeled = [X est_states]';
            plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(ts(i)),'), K:',num2str(K),', loglik:',num2str(loglik)), {'x_1','x_2'},label_range)
        end
    end
    
    % Segmentation Metrics per run
    [relabeled_est_states_all, hamming_distance(run),~,~] = mapSequence2Truth(true_states_all,est_states_all);
    [~,global_consistency(run), variation_info(run)] = compare_segmentations(true_states_all,est_states_all);
    
    % Cluster Metrics per run
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, relabeled_est_states_all);
    
end

% Overall Stats for HMM segmentation and state clustering
clc;
fprintf('*** Hidden Markov Model Results*** \n Optimal States: %d \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[K mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

% Visualize Transition Matrix and Segmentation from 'Best' Run
% Visualize Transition Matrix
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotTransMatrix(A);

% Visualize Estimated Emission Parameters
title_name  = 'Estimated Emission Parameters';
plot_labels = {'$x_1$','$x_2$'};
Est_theta.Mu = phi.mu;
Est_theta.Sigma = phi.Sigma;
Est_theta.K = K;
if exist('h2','var') && isvalid(h2), delete(h2);end
h2 = plotGaussianEmissions2D(Est_theta, plot_labels, title_name);

