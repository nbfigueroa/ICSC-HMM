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
[data, Data, True_states, True_theta] = genToyHMMData_Gaussian( N_TS, display ); 
super_states = 0;

%% 2a) Toy 2D dataset, 4 Unique Emission models, 5 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_Gaussian( 4, 2, 2, 500, 0.5 ); 
dataset_name = '2D';
super_states = 0;

% Feat matrix F (binary 5 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

%% 2b) Toy 2D dataset, 2 Unique Emission models transformed, 4 time-series
clc; clear all; close all;
[data, TruePsi, Data, True_states] = genToySeqData_TR_Gaussian(4, 2, 3, 500, 0.5 );
dataset_name = '2D Transformed'; 
super_states = 1;

% Feat matrix F (binary 4 x 4 matrix )
if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = plotFeatMat( TruePsi.F);

% Similarity matrix S (4 x 4 matrix)
if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = plotSimMat( TruePsi.S );

%% 3) Real 'Grating' 7D dataset, 3 Unique Emission models, 12 time-series
%Demonstration of a Carrot Grating Task consisting of 
%12 (7-d) time-series X = {x_1,..,x_T} with variable length T. 
%Dimensions:
%x = {pos_x, pos_y, pos_z, q_i, q_j, q_k, q_w}
clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'same'; full = 0; 
rf = 'same'; % Define if you want data recorded from the same reference frame or 'diff'
[data, ~, Data, True_states] = load_grating_dataset( data_path, type, display, full);
dataset_name = 'Grating'; Data_ = Data;  super_states = 0;

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
% type: 'proc', sub-sampled to 100 Hz, smoothed f/t trajectories, fixed rotation
% discontinuities.

clc; clear all; close all;
data_path = './test-data/'; display = 1; type = 'proc'; full = 0; 
% Type of data processing
% O: no data manipulation -- 1: zero-mean -- 2: scaled by range * weights
normalize = 2; 

% Define weights for dimensionality scaling
weights = [7*ones(1,3) ones(1,4) 1/10*ones(1,3) 0*ones(1,3)]';

% Define if using first derivative of pos/orient
use_vel = 1;
[data, TruePsi, ~, True_states, Data_] = load_rolling_dataset( data_path, type, display, full, normalize, weights, use_vel);
dataset_name = 'Rolling'; super_states = 0; 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Run Collapsed IBP-HMM Sampler T times for good statistics          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Settings for IBP-HMM %%%

% IBP hyper-parametrs
gamma = length(Data);  % length(Data)
alpha = 1;  % typically 1.. could change
kappa = 10; % sticky parameter

% Model Setting (IBP mass, IBP concentration, HMM alpha, HMM sticky)
modelP = {'bpM.gamma', gamma, 'bpM.c', 1, 'hmmM.alpha', alpha, 'hmmM.kappa', kappa}; 

% Sampler Settings
algP   = {'Niter', 500, 'HMM.doSampleHypers',1,'BP.doSampleMass',1,'BP.doSampleConc', 0, ...
         'doSampleFUnique', 1, 'doSplitMerge', 1} ;

% Number of Repetitions
T = 10; 

% Run MCMC Sampler for T times
Sampler_Stats = [];
jobID = ceil(rand*1000);
for run=1:T       
    % Run Gibbs Sampler for Niter once.
    clear CH    
    % Start out with just one feature for all objects
    initP  = {'F.nTotal', randsample(ceil(data.N),1)}; 
    CH = runBPHMM( data, modelP, {jobID, run}, algP, initP, './ibp-Results' );  
    Sampler_Stats(run).CH = CH;
end

%% %%%%%%%% Visualize Sampler Convergence/Metrics and extract Best Psi/run %%%%%%%%%%
if exist('h1','var')  && isvalid(h1),  delete(h1);end
if exist('h1b','var') && isvalid(h1b), delete(h1b);end
[h1, h1b, Best_Psi] = plotSamplerStatsBestPsi(Sampler_Stats, 'hist');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%      Run Collapsed SPCM-CRP Sampler on Theta        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
est_labels     = [];
clust_logProbs = [];
for l=1:length(Best_Psi)    
    % Extract info from 'Best Psi'
    K_est = Best_Psi(l).nFeats;
    
    % Extract Sigmas
    sigmas = [];
    for k=1:K_est
        invSigma = Best_Psi(l).Psi.theta(k).invSigma;
        sigmas{k} = invSigma \ eye(size(invSigma,1));
    end
    
    % Settings and Hyper-Params for SPCM-CRP Clustering algorithm
    clear clust_options
    clust_options.tau           = 1;       % Tolerance Parameter for SPCM-CRP
    clust_options.type          = 'full';  % Type of Covariance Matrix: 'full' = NIW or 'Diag' = NIG
    clust_options.T             = 100;     % Sampler Iterations
    clust_options.alpha         = 1;       % Concentration parameter
    clust_options.plot_sim      = 0;
    clust_options.init_clust    = 1:length(sigmas);
    clust_options.verbose       = 1;
    
    % Inference of SPCM-CRP Mixture Model
    [Psi Psi_Stats est_labels{l}]  = run_SPCMCRP_mm(sigmas, clust_options);
    clust_logProbs = [clust_logProbs Psi.MaxLogProb];
end

%% %%%% Compute Clustering/Segmentation Metrics vs Ground Truth %%%%%%
if isfield(TruePsi, 'sTrueAll')
    true_states_all = TruePsi.sTrueAll;
else
    true_states_all = data.zTrueAll;
end

% Compute metrics for IBP-HMM
[ results ] = computeSegmClustmetrics(true_states_all, Best_Psi, est_labels);

%% Choose best run
log_probs = zeros(1,T);
for ii=1:T; log_probs(ii) = Best_Psi(ii).logPr + clust_logProbs(ii); end

[val_max id_max] = sort(log_probs,'descend')

bestPsi      = Best_Psi(id_max(1));
est_labels_  = est_labels{id_max(1)};

%% Plot Segmentation+Clustering with Chosen Run and Metrics
if exist('h2','var') && isvalid(h2), delete(h2);end
[ h2 ] = plotDoubleLabelSegmentation(data, bestPsi, est_labels_);

% Plot Estimated Feature Matrix
if exist('h3','var') && isvalid(h3), delete(h3);end
[ h3 ] = plotFeatMat( bestPsi.Psi.F);

% Plot Estimated Transition Matrices
if exist('h4','var') && isvalid(h4), delete(h4);end
[h4, bestPsi] = plotTransitionMatrices(bestPsi);

% Compute Segmentation and State Clustering Metrics
results = computeSegmClustmetrics(true_states_all, bestPsi, est_labels_);

%% Plot Estimated  Emission Parameters for 2D Datasets ONLY!
title_name  = 'Estimated Emission Parameters';
plot_labels = {'$x_1$','$x_2$'};
clear Est_theta
Est_theta.K = K_est;
for k=1:K_est
    Est_theta.Mu(:,k)         = bestPsi.Psi.theta(k).mu;
    Est_theta.invSigma(:,:,k) = bestPsi.Psi.theta(k).invSigma;
    Est_theta.Sigma(:,:,k)    = Est_theta.invSigma(:,:,k) \ eye(data.D);
end

if exist('h4','var') && isvalid(h4), delete(h4);end
h4 = plotGaussianEmissions2D(Est_theta, plot_labels, title_name, est_labels_);

%% Visualize Segmented Trajectories in 3D ONLY!

% Plot Segmentated 3D Trajectories
labels    = unique(est_states_all);
titlename = strcat(dataset_name,' Demonstrations (Estimated Segmentation)');
if exist('h5','var') && isvalid(h5), delete(h5);end
h5 = plotLabeled3DTrajectories(Data_, est_states, titlename, labels);

% Plot Clustered/Segmentated 3D Trajectories
labels    = unique(est_clust_states_all);
titlename = strcat(dataset_name,' Demonstrations (Estimated Clustered-Segmentation)');
if exist('h6','var') && isvalid(h6), delete(h6);end
h6 = plotLabeled3DTrajectories(Data_, est_clust_states, titlename, labels);

% Plot Segmentated 3D Trajectories
titlename = strcat(dataset_name,' Demonstrations (Ground Truth)');
if exist('h7','var') && isvalid(h7), delete(h7);end
h7 = plotLabeled3DTrajectories(Data_, True_states, titlename, unique(data.zTrueAll));

