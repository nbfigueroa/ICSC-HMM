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
[Data, True_states] = genToyHMMData_Gaussian( N_TS, display ); 
label_range = unique(True_states{1});

%% 2) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data_nbp, TruePsi_nbp, Data, True_states] = genToySeqData_Gaussian( 4, 2, 5, 500, 0.5 ); 
label_range = unique(data_nbp.zTrueAll);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Sticky HDP-HMM Sampler T times for good statistics             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Default Setting for HDP-HMM
obsModelType = 'Gaussian';
priorType = 'NIW';
d = size(Data{1},2);
kappa = 0.1;           % NIW(kappa,theta,delta,nu_delta)
meanSigma = eye(d);    % expected mean of IW(nu,nu_delta) prior on Sigma_{k,j}
Kz = 10;               % truncation level of the DP prior on HMM transition distributions pi_k
Ks = 1;                % truncation level of the DPMM on emission distributions pi_s
saveDir = './Results';
plot_iter = 1;
Niter = 1000;

% Sampler Settings for HDP-HMM:
clear settings
settings.Kz = Kz;              % truncation level for mode transition distributions
settings.Ks = Ks;              % truncation level for mode transition distributions
settings.Niter = Niter;         % Number of iterations of the Gibbs sampler
settings.resample_kappa = 1;   % Whether or not to use sticky model
settings.seqSampleEvery = 100; % How often to run sequential z sampling
settings.saveEvery = 100;      % How often to save Gibbs sample stats
settings.storeEvery = 1;
settings.storeStateSeqEvery = 100;
settings.ploton = plot_iter;    % Plot the mode sequence while running sampler
settings.plotEvery = 20;
settings.plotpause = 0;        % Length of time to pause on the plot
settings.saveDir = saveDir;    % Directory to which to save files
settings.formZinit = 1;

% Dynamical Model Settings for HDP-HMM:
clear model
model.obsModel.type = obsModelType;
model.obsModel.priorType = priorType;
model.obsModel.params.M  = zeros([d 1]);
model.obsModel.params.K =  kappa;
% Degrees of freedom and scale matrix for covariance of process noise:
model.obsModel.params.nu = 1000; %d + 2;
model.obsModel.params.nu_delta = (model.obsModel.params.nu-d-1)*meanSigma;
      
% Always using DP mixtures emissions, with single Gaussian forced by
% Ks=1...Need to fix.
model.obsModel.mixtureType = 'infinite';

% Sticky HDP-HMM parameter settings:
model.HMMmodel.params.a_alpha=1;  % affects \pi_z
model.HMMmodel.params.b_alpha=0.01;
model.HMMmodel.params.a_gamma=1;  % global expected # of HMM states (affects \beta)
model.HMMmodel.params.b_gamma=0.01;
if settings.Ks>1
    model.HMMmodel.params.a_sigma = 1;
    model.HMMmodel.params.b_sigma = 0.01;
end
if isfield(settings,'Kr')
    if settings.Kr > 1
        model.HMMmodel.params.a_eta = 1;
        model.HMMmodel.params.b_eta = 0.01;
    end
end
model.HMMmodel.params.c=100;  % self trans
model.HMMmodel.params.d=1;
model.HMMmodel.type = 'HDP';

%% Run Weak-Limit Gibbs Sampler for sticky HDP-HMM
% Create data structure of multiple time-series for HDP-HMM sampler
clear data_struct 
for ii=1:length(Data)
    data_struct(ii).obs = Data{ii}';
%     data_struct(ii).true_labels = True_states{ii}';
end
data_struct(1).test_cases = [1:length(data_struct)];

% Options for sticky-HDP-HMM sampler, (1) Gaussian emission models
% hdp_options clear
% hdp_options.
%...

T = 10; % Number of Repetitions
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
    tic;
    clear ChainStats
    settings.trial = 1;
    [ChainStats] = HDPHMMDPinference(data_struct,model,settings);
    toc;    
    
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
        true_states = True_states{jj};
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

% Clean up metrics.. F-measure might give nan's sometimes

% Final Stats for Sticky HDP-HMM segmentation and state clustering
clc;
fprintf('*** Sticky HDP-HMM Results*** \n Optimal States: %3.3f (%3.3f) \n Hamming-Distance: %3.3f (%3.3f) GCE: %3.3f (%3.3f) VO: %3.3f (%3.3f) \n Purity: %3.3f (%3.3f) NMI: %3.3f (%3.3f) F: %3.3f (%3.3f)  \n',[mean(inferred_states) std(inferred_states) mean(hamming_distance) std(hamming_distance)  ...
    mean(global_consistency) std(global_consistency) mean(variation_info) std(variation_info) mean(cluster_purity) std(cluster_purity) mean(cluster_NMI) std(cluster_NMI) mean(cluster_F) std(cluster_F)])

%% Visualize Transition Matrix and Segmentation from 'Best' Run
log_likelihoods = zeros(1,T);
for ii=1:T; log_likelihoods(ii) = mean(ChainStats_Run(ii).logliks); end
[Max_ll, id] = max(log_likelihoods);
BestChain = ChainStats_Run(id);
K_est = inferred_states(id);
est_states_all = [];
for ii=1:length(data_struct); est_states_all  = [est_states_all BestChain.stateSeq(ii).z]; end
label_range = unique(est_states_all)

% Optimal Theta
Theta.mu    =  BestChain.Theta.mu(:,label_range);
Theta.Sigma =  BestChain.Theta.invSigma(:,:,label_range);

% Plot Segmentation
figure('Color',[1 1 1])
for i=1:length(data_struct)
    X = data_struct(i).obs;
    
    % Segmentation Direct from state sequence (Gives the same output as Viterbi estimate)
    est_states  = BestChain.stateSeq(i).z;
    
    % Plot Inferred Segments
    subplot(length(data_struct),1,i);
    data_labeled = [X; est_states];
    plotLabeledData( data_labeled, [], strcat('Segmented Time-Series (', num2str(i),'), K:',num2str(K_est),', loglik:',num2str(ChainStats_Run(2).logliks(i))), {'x_1','x_2'},label_range)
    
end

% Plot Transition Matrix
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotTransMatrix(BestChain.TransProb);

