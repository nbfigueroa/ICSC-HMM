%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main demo script for the ICSC-HMM Segmentation Algorithm proposed in:
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
[Data, True_states] = genToyHMMData_Gaussian( N_TS, display ); 

%% 2) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 

Data = []; True_states = [];
% Extract data for HMM
for i=1:data.N
    Data{i} = data.seq(i)';
    True_states{i} = data.zTrue(i)';
end
ts = [1:length(Data)];
figure('Color',[1 1 1])
for i=1:length(ts)
    X = Data{ts(i)};
    true_states = True_states{ts(i)};
    
    % Plot time-series with true labels
    subplot(length(ts),1,i);
    data_labeled = [X true_states]';
    plotLabeledData( data_labeled, [], strcat('Time-Series (', num2str(ts(i)),') with true labels'), {'x_1','x_2'})
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Run E-M Model Selection for HMM with 10 runs in a range of K     %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model Selection for HMM
K_range = [1:10]; repeats = 5; 
hmm_eval(Data, K_range, repeats)

%%  Fit HMM with 'optimal' K and Apply Viterbi for Segmentation
% Set "Optimal " GMM Hyper-parameters
K = 4; 
tic;
[p_start, A, phi, loglik] = ChmmGauss(Data, K);
toc;

% Calculate p(X) & viterbi decode fo reach time-series
ts = [1:length(Data)];
segmentation_hamm = zeros(1,length(ts));
segmentation_acc  = zeros(1,length(ts));
true_states_all   = [];
est_states_all    = [];

figure('Color',[1 1 1])
for i=1:length(ts)
    X = Data{ts(i)};
    true_states = True_states{ts(i)};
    logp_xn_given_zn = Gauss_logp_xn_given_zn(Data{i}, phi);
    [~,~, loglik] = LogForwardBackward(logp_xn_given_zn, p_start, A);
    est_states = LogViterbiDecode(logp_xn_given_zn, p_start, A);    
        
    % Stack all labels`
    true_states_all = [true_states_all; true_states ];
    est_states_all  = [est_states_all; est_states];
    
    % Segmentation Metrics
    [relabeled_est_states, segmentation_hamm(i),~,~] = mapSequence2Truth(true_states,est_states);    
    
    % Plot segmentation Results on each time-series
    subplot(length(ts),1,i);
    data_labeled = [X est_states]';
    plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(ts(i)),'), K:',num2str(K),', loglik:',num2str(loglik)), {'x_1','x_2'})
end

% Cluster Metrics
[cluster_purity cluster_NMI cluster_F] = cluster_metrics(true_states_all, est_states_all)