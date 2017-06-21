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
label_range = unique(True_states{1});

%% 2a) Toy 2D dataset, 4 Unique Emission models, max 5 time-series
clc; clear all; close all;
M = 4; % Number of Time-Series
[~, ~, Data, True_states] = genToySeqData_Gaussian( 4, 2, M, 500, 0.5 ); 

%% 2b) Toy 2D dataset, 2 Unique Emission models transformed, max 4 time-series
clc; clear all; close all;
M = 3; % Number of Time-Series
[~, TruePsi, Data, True_states] = genToySeqData_TR_Gaussian(4, 2, M, 500, 0.5 );
dataset_name = '2D Transformed'; 

% Similarity matrix S (4 x 4 matrix)
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSimMat( TruePsi.S );

%% 3) Real 'Grating' 7D dataset, 3 Unique Emission models, 12 time-series
%Demonstration of a Carrot Grating Task consisting of 
%12 (7-d) time-series X = {x_1,..,x_T} with variable length T. 
%Dimensions:
%x = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w}
clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'mixed'; full = 0; use_vel = 1;
[data, TruePsi, Data, True_states ,Data_] = load_grating_dataset( data_path, type, display, full, use_vel);
dataset_name = 'Grating'; 

%% 4) Real 'Dough-Rolling' 12D dataset, 3 Unique Emission models, 12 time-series
% Demonstration of a Dough Rolling Task consisting of 
% 15 (13-d) time-series X = {x_1,..,x_T} with variable length T. 
%
% Dimensions:
% x = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w, f_x, f_y, f_z, tau_x, tau_y, tau_z}
% - positions:         Data{i}(1:3,:)   (3-d: x, y, z)
% - orientations:      Data{i}(4:7,:)   (4-d: q_i, q_j, q_k, q_w)
% - forces:            Data{i}(8:10,:)   (3-d: f_x, f_y, f_z)
% - torques:           Data{i}(11:13,:) (3-d: tau_x, tau_y, tau_z)

% Dataset type:
%
% type: 'raw', raw sensor recordings at 500 Hz, f/t readings are noisy af and
% quaternions dimensions exhibit discontinuities
% This dataset is NOT labeled
%
% type: 'proc', sub-sampled to 100 Hz, smoothed f/t trajactories, fixed rotation
% discontinuities.

clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'proc'; full = 0; 
% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 2; 

% Define weights for dimensionality scaling
weights = [5*ones(1,3) ones(1,4) 1/10*ones(1,3) 0*ones(1,3)]';

% Define if using first derivative of pos/orient
use_vel = 1;
[~, ~, Data, True_states, Data_] = load_rolling_dataset( data_path, type, display, full, normalize, weights, use_vel);
dataset_name = 'Rolling'; super_states = 0;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Sticky HDP-HMM Sampler T times for good statistics             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for HDP-HMM %%%
clear hdp_options
hdp_options.obsModelType = 'Gaussian';
hdp_options.priorType = 'NIW';
hdp_options.d = size(Data{1},2);
hdp_options.sticky = 1;
hdp_options.kappa = 0.5;                    % NIW(kappa,theta,delta,nu_delta)
hdp_options.meanSigma = eye(hdp_options.d); % expected mean of IW(nu,nu_delta) prior on Sigma_{k,j}
hdp_options.Kz = 10;                        % truncation level of the DP prior on HMM transition distributions pi_k
hdp_options.Ks = 1;                         % truncation level of the DPMM on emission distributions pi_s (1-Gaussian emission)
hdp_options.plot_iter = 0;
hdp_options.Niter = 500;
hdp_options.saveDir = './hdp-Results';

%%% Create data structure of multiple time-series for HDP-HMM sampler %%%
clear data_struct 
for ii=1:length(Data)
    data_struct(ii).obs = Data{ii}';    
    % Set true_labels to visualize the sampler evolution
    % data_struct(ii).true_labels = True_states{ii}';
end

%%%% Run Weak-Limit Gibbs Sampler for sticky HDP-HMM %%%
% Number of Repetitions
T = 10;

% Segmentation Metric Arrays
hamming_distance   = zeros(1,T);
global_consistency = zeros(1,T);
variation_info     = zeros(1,T);
inferred_states    = zeros(1,T);

% Clustering Metric Arrays
cluster_purity = zeros(1,T);
cluster_NMI    = zeros(1,T);
cluster_F      = zeros(1,T);

% Run Weak Limit Collapsed Gibbs Sampler for T times
ChainStats_Run = [];
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear ChainStats
    ChainStats = run_HDPHMM_sampler(data_struct, hdp_options);
    
    % Extract Stats from Last Run
    ChainStats_Run(run).stateSeq  = ChainStats(end).stateSeq;
    ChainStats_Run(run).initProb  = ChainStats(end).dist_struct(end).pi_init;
    ChainStats_Run(run).TransProb = ChainStats(end).dist_struct(end).pi_z;
    ChainStats_Run(run).Theta     = ChainStats(end).theta(end);
    
    % Compute Log-Likelihood and est_states for this run
    true_states_all   = [];
    est_states_all    = [];
    loglik_ = zeros(length(data_struct),1);
    Theta.mu =  ChainStats_Run(run).Theta.mu;
    Theta.Sigma =  ChainStats_Run(run).Theta.invSigma;
    for jj=1:length(data_struct)
        logp_xn_given_zn = Gauss_logp_xn_given_zn(data_struct(jj).obs', Theta);
        [~,~, loglik_(jj,1)] = LogForwardBackward(logp_xn_given_zn, ChainStats_Run(run).initProb, ChainStats_Run(run).TransProb);
        
        % Stack labels for state clustering metrics   
        if isfield(TruePsi, 'sTrueAll')
            true_states = TruePsi.s{jj}';
        else
            true_states = True_states{jj};
        end
        
        est_states  = ChainStats_Run(run).stateSeq(jj).z';
        true_states_all = [true_states_all; true_states];
        est_states_all  = [est_states_all; est_states];
        
    end
    ChainStats_Run(run).logliks = loglik_;    
    
    % Segmentation Metrics per run
    [relabeled_est_states_all, hamming_distance(run),~,~] = mapSequence2Truth(true_states_all,est_states_all);
    [~,global_consistency(run), variation_info(run)] = compare_segmentations(true_states_all,est_states_all);
    inferred_states(run)   = length(unique(est_states_all));
    
    % Cluster Metrics per run
    [cluster_purity(run) cluster_NMI(run) cluster_F(run)] = cluster_metrics(true_states_all, relabeled_est_states_all);        
end

% Overall Stats for HMM segmentation and state clustering
fprintf('*** Sticky HDP-HMM Results*** \n Optimal States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

%% Visualize Transition Matrix, Emission Parameters and Segmentation from 'Best' Run
mean_likelihoods = zeros(1,T);
std_likelihoods = zeros(1,T);
for ii=1:T 
    mean_likelihoods(ii) = mean(ChainStats_Run(ii).logliks); 
    std_likelihoods(ii) = std(ChainStats_Run(ii).logliks); 
end
% [Max_ll, max_id] = max(log_likelihoods)
[val_std id_std] = sort(std_likelihoods,'ascend');
[val_mean id_mean] = sort(mean_likelihoods,'descend');

id_mean
id_std

%% Plot Segmentation with Chosen Run
id = id_mean(2);
BestChain = ChainStats_Run(id);
K_est = inferred_states(id);
est_states_all = [];
for ii=1:length(data_struct); est_states_all  = [est_states_all BestChain.stateSeq(ii).z]; end
label_range = unique(est_states_all);
est_states = [];

% Plot Segmentation and Stats
figure('Color',[1 1 1])
true_states_all   = [];
est_states_all    = [];

for i=1:length(data_struct)
    X = data_struct(i).obs;
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states{i}  = BestChain.stateSeq(i).z;
    
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
    plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(i),'), K:',num2str(K_est),', loglik:',num2str(ChainStats_Run(2).logliks(i))), [],label_range)
    
end

% Segmentation Metrics per run
[relabeled_est_states_all, hamming_distance_,~,~] = mapSequence2Truth(true_states_all,est_states_all);
[~,global_consistency_, variation_info_] = compare_segmentations(true_states_all,est_states_all);
inferred_states_   = length(unique(est_states_all));

% Cluster Metrics per run
[cluster_purity_ cluster_NMI_ cluster_F_] = cluster_metrics(true_states_all, relabeled_est_states_all);

% Overall Stats for HMM segmentation and state clustering
fprintf('*** Sticky HDP-HMM Results*** \n Optimal States: %3.3f \n Hamming-Distance: %3.3f GCE: %3.3f VO: %3.3f \n Purity: %3.3f  NMI: %3.3f  F: %3.3f   \n',[inferred_states_  hamming_distance_  ...
    global_consistency_ variation_info_ cluster_purity_ cluster_NMI_ cluster_F_])

% Plot Transition Matrix
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotTransMatrix(BestChain.TransProb);

%% Visualize Estimated Emission Parameters for 2D Data ONLY!
title_name  = 'Estimated Emission Parameters';
plot_labels = {'$x_1$','$x_2$'};
clear Est_theta
Est_theta.Mu = BestChain.Theta.mu(:,label_range);
Est_theta.invSigma = BestChain.Theta.invSigma(:,:,label_range);
Est_theta.K = K_est;
for k=1:K_est
    Est_theta.Sigma(:,:,k) = Est_theta.invSigma(:,:,k) \ eye(hdp_options.d);
end

if exist('h2','var') && isvalid(h2), delete(h2);end
h2 = plotGaussianEmissions2D(Est_theta, plot_labels, title_name);

%% Visualize Segmented Trajectories in 3D ONLY!
labels    = unique(est_states_all);
titlename = strcat(dataset_name,' Demonstrations (Estimated Segmentation)');

% Plot Segmentated 3D Trajectories
if exist('h5','var') && isvalid(h5), delete(h5);end
h5 = plotLabeled3DTrajectories(Data_, est_states, titlename, labels);

titlename = strcat(dataset_name,' Demonstrations (Ground Truth)');
% Plot Segmentated 3D Trajectories
if exist('h6','var') && isvalid(h6), delete(h6);end
h6 = plotLabeled3DTrajectories(Data_, True_states, titlename, [1:3]);
